"""Enode Backtester - Event-driven backtesting framework.

A lightweight, extensible backtesting framework that provides:
- Event-driven architecture preventing look-ahead bias
- Realistic execution simulation with slippage and commissions
- Modular design for easy strategy development and testing
- Comprehensive performance analysis and reporting

Quick Start:
    import pandas as pd
    from backtester import run_backtest, FixedSizeSizer
    from my_strategies import MyStrategy
    
    # Prepare data
    data_dict = {"AAPL": aapl_df, "GOOGL": googl_df}
    
    # Run backtest
    portfolio = run_backtest(
        data_dict=data_dict,
        strategy_class=MyStrategy,
        initial_cash=100_000.0,
        sizer=FixedSizeSizer(100)
    )
    
    # Analyze results
    print(f"Final equity: ${portfolio.equity_curve[-1][1]:,.2f}")

Main Components:
    - BaseStrategy: Implement your trading logic
    - Portfolio: Manages positions and cash
    - DataFrameDataHandler: Streams historical data
    - SimulatedExecutionHandler: Realistic order execution
    - BacktestEngine: Coordinates the event loop
"""

import queue
from typing import Type, Callable, Optional

import pandas as pd

from .analysis import generate_full_tear_sheet, get_trade_log_df
from .metrics import analyze_strategy, StrategyAnalyzer, PerformanceMetrics

class BacktestResult:
    """Wrapper for backtest results with convenience methods."""
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self._metrics = None
        self._monte_carlo = None
    
    @property
    def metrics(self):
        """Lazy-load performance metrics."""
        if self._metrics is None:
            self._metrics, self._monte_carlo = analyze_strategy(self.portfolio)
        return self._metrics
    
    @property
    def monte_carlo(self):
        """Lazy-load Monte Carlo analysis."""
        if self._monte_carlo is None:
            self._metrics, self._monte_carlo = analyze_strategy(self.portfolio)
        return self._monte_carlo
    
    def dashboard(self, port: int = 8050, save: bool = False):
        """Launch interactive dashboard."""
        if not _check_dashboard():
            raise ImportError("Dashboard requires: pip install dash plotly dash-bootstrap-components")
        
        if save:
            self.save("backtest_results.json")
        
        launch_dashboard(portfolio=self.portfolio, port=port)
        return self
    
    def save(self, filename: str = "backtest_results.json"):
        """Save results to JSON file."""
        from .dashboard.loaders import save_results
        save_results(self.portfolio, self.metrics, self.monte_carlo, filename)
        print(f"💾 Results saved to {filename}")
        return self
    
    def __getattr__(self, name):
        """Delegate to portfolio for backward compatibility."""
        return getattr(self.portfolio, name)

# Dashboard availability flag (lazy import)
_DASHBOARD_AVAILABLE = None
create_app = None
launch_dashboard = None

def _check_dashboard():
    """Lazy check for dashboard availability."""
    global _DASHBOARD_AVAILABLE, create_app, launch_dashboard
    if _DASHBOARD_AVAILABLE is None:
        try:
            from .dashboard import create_app, launch_dashboard
            _DASHBOARD_AVAILABLE = True
        except ImportError:
            _DASHBOARD_AVAILABLE = False
            create_app = None
            launch_dashboard = None
    return _DASHBOARD_AVAILABLE
from .data import DataFrameDataHandler
from .engine import BacktestEngine
from .execution import (
    SimulatedExecutionHandler,
    default_commission_model,
    default_slippage_model,
)
from .portfolio import Portfolio
from .sizer import BaseSizer, FixedSizeSizer
from .strategy import BaseStrategy
from .risk import (
    RiskManager,
    BaseRiskRule,
    MaxPositionSizeRule,
    MaxCashUsageRule,
    MaxPositionCountRule,
    MaxDrawdownRule,
    StopLossRule,
)


def run_backtest(
    data_dict: dict[str, pd.DataFrame],
    strategy_class: Type[BaseStrategy],
    initial_cash: float,
    sizer: Optional[BaseSizer] = None,
    risk_manager: Optional[RiskManager] = None,
    commission_model: Optional[Callable] = None,
    slippage_model: Optional[Callable] = None,
) -> BacktestResult:
    """Execute a complete backtesting simulation.
    
    This is the main entry point for running backtests. It instantiates all
    required components, executes the simulation, and returns the final portfolio
    state for analysis.
    
    Args:
        data_dict: Dictionary mapping symbol names to pandas DataFrames.
                  Each DataFrame must contain columns: symbol, timestamp,
                  open, high, low, close, volume (volume optional)
        strategy_class: Class (not instance) that inherits from BaseStrategy.
                       Will be instantiated with (event_queue, data_handler)
        initial_cash: Starting capital in dollars
        sizer: Position sizing strategy. Defaults to FixedSizeSizer(50)
        risk_manager: Optional risk management system with rules
        commission_model: Function(quantity, fill_price) -> commission_cost.
                         Defaults to $0.005 per share
        slippage_model: Function(order, market_price) -> execution_price.
                       Defaults to 0.01% slippage
    
    Returns:
        Portfolio: Final portfolio state containing:
            - equity_curve: List of (timestamp, total_value) tuples
            - trade_log: List of FillEvent objects (all executed trades)
            - current_positions: Dict of symbol -> share_count
            - current_cash: Remaining cash balance
    
    Example:
        # Basic usage
        portfolio = run_backtest(
            data_dict={"AAPL": aapl_df},
            strategy_class=MovingAverageStrategy,
            initial_cash=100_000.0
        )
        
        # With custom configuration
        portfolio = run_backtest(
            data_dict=multi_symbol_data,
            strategy_class=MyStrategy,
            initial_cash=500_000.0,
            sizer=FixedSizeSizer(200),
            commission_model=lambda qty, price: max(1.0, qty * 0.001),
            slippage_model=lambda order, price: price * (1.0005 if order.signal == 'buy' else 0.9995)
        )
    
    Note:
        The function handles all component initialization and wiring.
        All components share the same event_queue for proper communication.
        
        Data Requirements:
        - DataFrames must be sorted by timestamp (handled automatically)
        - Timestamps should be timezone-aware or consistent
        - Missing data is handled gracefully (symbols with no data are skipped)
        
        The backtest runs until all historical data is exhausted across
        all symbols in data_dict.
    """
    print("--- Initializing Backtest Components ---")

    # Create shared event queue for inter-component communication
    event_queue = queue.Queue()
    
    # Initialize data handler to stream historical market data
    data_handler = DataFrameDataHandler(event_queue, data_dict)
    
    # Instantiate the trading strategy
    strategy = strategy_class(event_queue, data_handler)
    
    # Set up portfolio with position tracking and cash management
    portfolio = Portfolio(
        event_queue,
        data_handler,
        initial_cash,
        sizer=sizer or FixedSizeSizer(50),
        risk_manager=risk_manager,
    )

    # Configure execution handler with cost models
    execution_handler = SimulatedExecutionHandler(
        event_queue,
        data_handler,
        commission_model=commission_model or default_commission_model,
        slippage_model=slippage_model or default_slippage_model,
    )
    
    # Create engine to coordinate the simulation
    engine = BacktestEngine(event_queue, data_handler, strategy, portfolio, execution_handler)

    # Execute the backtest
    final_portfolio = engine.run()

    print("--- Backtest Complete. Returning results. ---")
    return BacktestResult(final_portfolio)


# Version
__version__ = "0.1.3"

# Public API exports
__all__ = [
    "run_backtest",
    "generate_full_tear_sheet", 
    "get_trade_log_df",
    "analyze_strategy",
    "StrategyAnalyzer",
    "PerformanceMetrics",
    "create_app",
    "launch_dashboard",
    "FixedSizeSizer",
    "BaseStrategy",
    "BaseSizer",
    "RiskManager",
    "BaseRiskRule",
    "MaxPositionSizeRule",
    "MaxCashUsageRule",
    "MaxPositionCountRule",
    "MaxDrawdownRule",
    "StopLossRule",
]