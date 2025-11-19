"""Portfolio management module for tracking positions, cash, and performance.

This module manages the portfolio state throughout the backtest, including:
- Cash and position tracking
- Converting trading signals into orders
- Processing trade executions
- Recording equity curve and trade history
- Risk management and order validation
"""

import queue
from datetime import datetime
from collections import defaultdict
from typing import Optional

from .event import StockEvent, SignalEvent, OrderEvent, FillEvent
from .data import BaseDataHandler
from .sizer import BaseSizer, FixedSizeSizer
from .risk import RiskManager 


class BasePortfolio(object):
    """Abstract base class for portfolio implementations.
    
    Provides a foundation for different portfolio management approaches
    that might include additional features like:
    - Risk management rules
    - Position limits
    - Margin trading
    - Multi-currency support
    """
    pass


class Portfolio(BasePortfolio):
    """Core portfolio management class for backtesting.
    
    The Portfolio is the central hub for managing trading capital and positions.
    It serves multiple critical functions:
    
    1. **Signal Processing**: Converts strategy signals into sized orders
    2. **Position Tracking**: Maintains current holdings for each symbol
    3. **Cash Management**: Tracks available cash and transaction costs
    4. **Performance Recording**: Maintains equity curve and trade history
    5. **Risk Management**: Can implement position limits and risk controls
    
    The Portfolio operates in the event loop by:
    - Receiving StockEvents to update position valuations
    - Receiving SignalEvents to generate orders
    - Receiving FillEvents to update positions and cash
    
    Attributes:
        event_queue: Central event queue for inter-component communication
        data_handler: Interface to current market data
        sizer: Position sizing strategy
        initial_cash: Starting capital amount
        current_cash: Available cash for trading
        current_positions: Number of shares held per symbol
        current_holdings_value: Market value of positions per symbol
        equity_curve: Time series of total portfolio value
        trade_log: History of all executed trades
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        data_handler: BaseDataHandler,
        initial_cash: float, 
        sizer: BaseSizer = FixedSizeSizer(50),
        risk_manager: Optional[RiskManager] = None
    ):
        """Initialize the portfolio with starting capital and configuration.
        
        Args:
            event_queue: Central event queue for the backtesting system
            data_handler: Interface to access current market data
            initial_cash: Starting capital amount in dollars
            sizer: Position sizing strategy (defaults to 50 shares per trade)
            risk_manager: Optional risk management system for order validation
        """
        self.event_queue = event_queue
        self.data_handler = data_handler
        self.sizer = sizer
        self.risk_manager = risk_manager
        self.initial_cash = initial_cash

        # Portfolio state
        self.current_cash = initial_cash
        self.current_positions = defaultdict(int)  # symbol -> share_count
        self.current_holdings_value = defaultdict(float)  # symbol -> market_value

        # Performance tracking
        self.equity_curve = []  # [(timestamp, total_equity), ...]
        self.trade_log = []  # [FillEvent, ...]
        self._last_equity_timestamp = None  # Track last recorded timestamp

    def on_stock_event(self, event: StockEvent) -> None:
        """Update position valuations and record equity when new market data arrives.
        
        This method is called for every new price update and serves two purposes:
        1. Revalue existing positions at current market prices
        2. Record the current total portfolio value for performance tracking
        
        Args:
            event: StockEvent containing new market data (OHLCV candle)
        
        Note:
            Only positions with non-zero holdings are revalued to avoid
            unnecessary calculations. The equity curve is updated on every
            price tick to capture intraday portfolio performance.
        """
        symbol = event.payload.symbol 

        # Update market value of existing positions
        if self.current_positions[symbol] != 0: 
            latest_price = event.payload.close
            market_value = latest_price * self.current_positions[symbol]
            self.current_holdings_value[symbol] = market_value

        # Record current total portfolio value
        self.record_equity(event.payload.timestamp)

    def record_equity(self, timestamp: datetime) -> None:
        """Calculate and record total portfolio equity at a given timestamp.
        
        Total equity = Available Cash + Market Value of All Positions
        
        This creates the equity curve used for performance analysis,
        including returns calculation, drawdown analysis, and visualization.
        
        Args:
            timestamp: When this equity measurement was taken
        
        Note:
            Only records equity once per unique timestamp to avoid duplicates
            when multiple symbols update on the same date.
        """
        # Only record equity once per timestamp
        if self._last_equity_timestamp != timestamp:
            total_holdings = sum(self.current_holdings_value.values())
            total_equity = self.current_cash + total_holdings
            self.equity_curve.append((timestamp, total_equity))
            self._last_equity_timestamp = timestamp

    def on_signal_event(self, event: SignalEvent) -> None:
        """Convert a trading signal into a sized order with risk validation.
        
        This method bridges the gap between strategy intent (SignalEvent)
        and executable orders (OrderEvent). The process involves:
        1. Using the Sizer to determine appropriate position size
        2. Validating the order through the RiskManager (if configured)
        3. Creating an OrderEvent with the approved/modified quantity
        4. Emitting the order for execution
        
        Args:
            event: SignalEvent containing trading intent (symbol, direction, timestamp)
        
        Note:
            The risk manager can:
            - Approve the order as-is
            - Reduce the order quantity to meet risk limits
            - Reject the order entirely
            
            Rejected orders are logged but not executed.
        """
        # Determine position size using the configured sizer
        quantity = self.sizer.size_order(self, event) 
        
        if quantity == 0:
            return  # Sizer decided not to trade

        # Apply risk management if configured
        if self.risk_manager is not None:
            risk_result = self.risk_manager.evaluate_order(
                self, event, quantity, self.data_handler
            )
            
            if not risk_result.approved:
                print(f"Order rejected by risk manager: {risk_result.reason}")
                return
            
            if risk_result.modified_quantity is not None:
                quantity = risk_result.modified_quantity
                if risk_result.reason:
                    print(f"Order modified by risk manager: {risk_result.reason}")
        
        # Create order for valid signal types
        if event.signal in ['buy', 'sell']:
            order_event = OrderEvent(
                symbol=event.symbol,
                signal=event.signal,
                timestamp=event.timestamp,
                quantity=quantity
            )
            self.event_queue.put(order_event)

    def on_fill_event(self, event: FillEvent) -> None:
        """Process a trade execution and update portfolio state.
        
        This method handles the final step of the trading process by:
        1. Updating cash based on trade proceeds and costs
        2. Updating position quantities
        3. Revaluing the affected position at current market price
        4. Recording the trade in the trade log
        
        Args:
            event: FillEvent containing execution details (price, quantity, commission)
        
        Note:
            The cash calculation includes both the trade proceeds and commission costs:
            - Buy: cash decreases by (price × quantity + commission)
            - Sell: cash increases by (price × quantity - commission)
            
            Position updates are additive:
            - Buy: increases position by quantity
            - Sell: decreases position by quantity (can go negative for short positions)
        """
        # Update cash and positions based on trade direction
        if event.signal == 'buy':
            self.current_cash -= (event.fill_price * event.quantity) + event.commission
            self.current_positions[event.symbol] += event.quantity
        elif event.signal == 'sell':
            self.current_cash += (event.fill_price * event.quantity) - event.commission
            self.current_positions[event.symbol] -= event.quantity

        # Revalue the position at current market price
        latest_price = self.data_handler.get_latest_bar_value(event.symbol, 'close')
        if latest_price is not None:
            market_value = self.current_positions[event.symbol] * latest_price
            self.current_holdings_value[event.symbol] = market_value
        
        # Record the trade for analysis
        self.trade_log.append(event)
    
    def dashboard(self, port: int = 8050, save_results: bool = False):
        """Launch interactive dashboard for this portfolio.
        
        Args:
            port: Port to run dashboard on
            save_results: Whether to save results to JSON file
        """
        from .dashboard.app import launch_dashboard
        
        if save_results:
            from .dashboard.loaders import save_results as save_fn
            from .metrics import analyze_strategy
            
            metrics, monte_carlo = analyze_strategy(self)
            save_fn(self, metrics, monte_carlo, "backtest_results.json")
            print("💾 Results saved to backtest_results.json")
        
        launch_dashboard(portfolio=self, port=port)