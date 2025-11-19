"""Performance analysis and reporting module.

This module provides tools for analyzing backtesting results, including:
- Converting portfolio data to standard formats
- Generating performance reports and tear sheets
- Calculating trading statistics and metrics
- Exporting results for further analysis
"""

from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .portfolio import Portfolio

try:
    import pyfolio as pf  # Optional dependency: pip install pyfolio
except ImportError:  # pragma: no cover - optional dependency
    pf = None 


def get_returns_from_equity_curve(equity_curve: list) -> pd.Series:
    """Convert portfolio equity curve to daily percentage returns.
    
    Transforms the portfolio's equity curve (list of timestamp-value pairs)
    into a pandas Series of daily percentage returns suitable for performance
    analysis tools like PyFolio.
    
    Args:
        equity_curve: List of (timestamp, total_value) tuples from Portfolio
    
    Returns:
        pd.Series: Daily percentage returns with datetime index
    
    Processing Steps:
        1. Convert list to DataFrame with proper datetime index
        2. Ensure timezone-naive timestamps (PyFolio requirement)
        3. Resample to daily frequency (handles intraday data)
        4. Forward-fill missing days (weekends, holidays)
        5. Calculate percentage returns
    
    Example:
        equity_curve = [(datetime(2024,1,1), 100000), (datetime(2024,1,2), 101000)]
        returns = get_returns_from_equity_curve(equity_curve)
        # Returns: Series with 0.01 (1%) return for 2024-01-02
    
    Note:
        Returns are calculated as: (value_t / value_t-1) - 1
        The first return is always 0 (no previous value to compare)
    """
    if not equity_curve:
        return pd.Series(dtype=float)

    # Convert to DataFrame with proper structure
    df = pd.DataFrame(equity_curve, columns=['timestamp', 'total_value'])
    
    # Set up datetime index (required by analysis tools)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Ensure timezone-naive index (PyFolio requirement)
    df.index = df.index.tz_localize(None)

    # Resample to daily frequency (handles intraday data)
    # Takes the last value of each day
    df_daily = df.resample('D').last()

    # Forward-fill missing days (weekends, holidays)
    df_daily = df_daily.ffill()

    # Calculate daily percentage returns
    returns_series = df_daily['total_value'].pct_change().fillna(0)
    
    return returns_series


def generate_full_tear_sheet(portfolio: "Portfolio", benchmark_returns: pd.Series = None) -> None:
    """Generate comprehensive performance report using PyFolio.
    
    Creates a detailed performance analysis including:
    - Returns and risk statistics
    - Drawdown analysis
    - Rolling performance metrics
    - Factor exposure analysis (if benchmark provided)
    - Performance attribution charts
    
    Args:
        portfolio: Final Portfolio object from backtest execution
        benchmark_returns: Optional benchmark returns for comparison
                          (e.g., S&P 500 returns for equity strategies)
    
    Note:
        Requires PyFolio installation: pip install pyfolio
        
        The tear sheet includes:
        - Summary statistics (Sharpe ratio, volatility, max drawdown)
        - Rolling performance charts
        - Drawdown periods analysis
        - Monthly/yearly returns heatmap
        - Underwater plot showing drawdown periods
        
        If benchmark_returns is provided, additional analysis includes:
        - Beta and alpha calculations
        - Factor exposure analysis
        - Relative performance metrics
    
    Example:
        # Basic tear sheet
        generate_full_tear_sheet(final_portfolio)
        
        # With benchmark comparison
        spy_returns = get_spy_returns()  # Your benchmark data
        generate_full_tear_sheet(final_portfolio, spy_returns)
    """
    # Convert portfolio equity curve to returns
    returns = get_returns_from_equity_curve(portfolio.equity_curve)
    
    if returns.empty:
        print("Equity curve is empty. Cannot generate report.")
        return

    if pf is None:
        print("PyFolio is not installed; skipping tear sheet generation.")
        print("Install with: pip install pyfolio")
        return

    print("--- Generating Performance Report (this may take a moment) ---")
    
    # Generate comprehensive performance report
    # This creates multiple charts and statistical analyses
    pf.create_full_tear_sheet(
        returns,
        benchmark_rets=benchmark_returns
    )
    print("--- Report Generation Complete ---")


def get_trade_log_df(portfolio: "Portfolio") -> pd.DataFrame:
    """Convert portfolio trade log to pandas DataFrame for analysis.
    
    Transforms the portfolio's trade_log (list of FillEvent objects) into
    a structured DataFrame suitable for trade analysis, performance attribution,
    and strategy evaluation.
    
    Args:
        portfolio: Portfolio object containing trade_log of FillEvent objects
    
    Returns:
        pd.DataFrame: Trade history with columns:
            - symbol: Stock symbol traded
            - quantity: Number of shares (positive for both buy/sell)
            - fill_price: Actual execution price
            - commission: Transaction cost
            - signal: Trade direction ('buy' or 'sell')
            Index: timestamp (when trade was executed)
    
    Example Usage:
        trades_df = get_trade_log_df(final_portfolio)
        
        # Analyze trading activity
        print(f"Total trades: {len(trades_df)}")
        print(f"Total commissions: ${trades_df['commission'].sum():.2f}")
        
        # Calculate trade P&L
        buy_trades = trades_df[trades_df['signal'] == 'buy']
        sell_trades = trades_df[trades_df['signal'] == 'sell']
        
        # Group by symbol for per-stock analysis
        by_symbol = trades_df.groupby('symbol').agg({
            'quantity': 'sum',  # Net position change
            'commission': 'sum',  # Total costs
            'fill_price': 'mean'  # Average price
        })
    
    Note:
        The DataFrame is sorted chronologically and indexed by timestamp.
        This enables time-series analysis of trading patterns and costs.
        
        For P&L calculation, you'll need to match buy/sell pairs or
        calculate position-weighted average costs.
    """
    if not portfolio.trade_log:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            'symbol', 'timestamp', 'quantity', 
            'fill_price', 'commission', 'signal'
        ])
    
    # Convert FillEvent objects to DataFrame
    # vars() extracts all attributes from dataclass instances
    trade_df = pd.DataFrame([vars(fill) for fill in portfolio.trade_log])
    
    # Set up proper datetime index and sorting
    trade_df['timestamp'] = pd.to_datetime(trade_df['timestamp'])
    trade_df = trade_df.set_index('timestamp').sort_index()
    
    return trade_df