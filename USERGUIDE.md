# Enode Backtester User Guide

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Creating Trading Strategies](#creating-trading-strategies)
5. [Data Preparation](#data-preparation)
6. [Running Backtests](#running-backtests)
7. [Risk Management](#risk-management)
8. [Position Sizing](#position-sizing)
9. [Performance Analysis](#performance-analysis)
10. [Interactive Dashboard](#interactive-dashboard)
11. [Command Line Interface](#command-line-interface)
12. [Advanced Features](#advanced-features)
13. [Best Practices](#best-practices)

---

## Overview

The Enode Backtester is a professional-grade, event-driven backtesting framework designed for quantitative trading strategy development and analysis. It provides:

- **Event-driven architecture** that prevents look-ahead bias
- **Realistic execution simulation** with slippage and commissions
- **Comprehensive risk management** system
- **Advanced performance analytics** with statistical rigor
- **Interactive dashboard** for visual analysis
- **Modular design** for easy customization and extension

### Key Architecture Components

The framework follows an event-driven architecture where components communicate through events:

```
Market Data → Strategy → Signals → Portfolio → Orders → Execution → Fills → Portfolio
```

**Core Components:**
- **Strategy**: Your trading logic that generates buy/sell signals
- **Portfolio**: Manages positions, cash, and converts signals to orders
- **DataHandler**: Streams historical data chronologically
- **ExecutionHandler**: Simulates realistic order execution
- **RiskManager**: Validates orders against risk rules
- **Engine**: Coordinates the event loop

---

## Installation

### Using uv (Recommended)
```bash
git clone https://github.com/CalinQuant22/enode-backtester
cd enode-backtester
uv sync
```

### Using pip
```bash
git clone https://github.com/CalinQuant22/enode-backtester
cd enode-backtester
pip install -r requirements.txt
```

### Optional Dependencies
For the interactive dashboard:
```bash
pip install dash plotly dash-bootstrap-components
```

---

## Quick Start

Here's a minimal example to get you started:

```python
import pandas as pd
import numpy as np
from backtester import run_backtest, FixedSizeSizer
from backtester.strategy import BaseStrategy
from backtester.event import StockEvent

# 1. Create a simple strategy
class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self, event_queue, data_handler):
        super().__init__(event_queue, data_handler)
        self.bought = False
    
    def on_stock_event(self, event: StockEvent) -> None:
        if not self.bought:
            self.signal(event.payload.symbol, event.payload.timestamp, 'buy')
            self.bought = True

# 2. Prepare sample data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
np.random.seed(42)
prices = 100 * (1 + np.random.normal(0.001, 0.02, len(dates))).cumprod()

data = pd.DataFrame({
    'timestamp': dates,
    'symbol': 'AAPL',
    'open': prices,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'close': prices,
    'volume': 1000000
})

# 3. Run backtest
results = run_backtest(
    data_dict={'AAPL': data},
    strategy_class=BuyAndHoldStrategy,
    initial_cash=100_000.0,
    sizer=FixedSizeSizer(100)
)

# 4. Analyze results
print(f"Total Return: {results.metrics.total_return:.2%}")
print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")

# 5. Launch dashboard
results.dashboard()
```

---

## Creating Trading Strategies

### Strategy Base Class

All strategies must inherit from `BaseStrategy` and implement the `on_stock_event` method:

```python
from backtester.strategy import BaseStrategy
from backtester.event import StockEvent

class MyStrategy(BaseStrategy):
    def __init__(self, event_queue, data_handler):
        super().__init__(event_queue, data_handler)
        # Initialize your strategy state here
        
    def on_stock_event(self, event: StockEvent) -> None:
        # Your trading logic here
        # Access price data: event.payload.close, event.payload.high, etc.
        # Generate signals: self.signal(symbol, timestamp, 'buy'/'sell')
        pass
```

### Strategy Development Pattern

1. **Initialize State**: Set up indicators, price history, and parameters
2. **Process Market Data**: Update indicators and internal state
3. **Generate Signals**: Emit buy/sell signals when conditions are met
4. **Maintain Position Awareness**: Track your current market position

### Example: Moving Average Crossover Strategy

```python
from collections import defaultdict
import numpy as np
from backtester.strategy import BaseStrategy
from backtester.event import StockEvent

class MovingAverageCrossStrategy(BaseStrategy):
    def __init__(self, event_queue, data_handler, short_window=10, long_window=30):
        super().__init__(event_queue, data_handler)
        self.short_window = short_window
        self.long_window = long_window
        
        # Store price history for each symbol
        self.prices = defaultdict(list)
        # Track current position to avoid duplicate signals
        self.position = defaultdict(lambda: 'FLAT')  # 'FLAT', 'LONG'

    def on_stock_event(self, event: StockEvent):
        symbol = event.payload.symbol
        price = event.payload.close
        timestamp = event.payload.timestamp

        # 1. Update price history
        self.prices[symbol].append(price)

        # 2. Wait for sufficient data
        if len(self.prices[symbol]) < self.long_window:
            return

        # 3. Calculate moving averages
        prices = self.prices[symbol]
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])

        # 4. Generate signals
        current_pos = self.position[symbol]

        # Buy when short MA crosses above long MA
        if short_ma > long_ma and current_pos == 'FLAT':
            self.signal(symbol, timestamp, 'buy')
            self.position[symbol] = 'LONG'

        # Sell when short MA crosses below long MA
        elif short_ma < long_ma and current_pos == 'LONG':
            self.signal(symbol, timestamp, 'sell')
            self.position[symbol] = 'FLAT'
```

### Advanced Strategy Features

#### Using Technical Indicators
```python
class RSIStrategy(BaseStrategy):
    def __init__(self, event_queue, data_handler, rsi_period=14):
        super().__init__(event_queue, data_handler)
        self.rsi_period = rsi_period
        self.prices = defaultdict(list)
    
    def calculate_rsi(self, prices):
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return None
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def on_stock_event(self, event: StockEvent):
        symbol = event.payload.symbol
        price = event.payload.close
        
        self.prices[symbol].append(price)
        
        rsi = self.calculate_rsi(self.prices[symbol])
        if rsi is None:
            return
            
        # RSI oversold/overbought strategy
        if rsi < 30:  # Oversold - buy signal
            self.signal(symbol, event.payload.timestamp, 'buy')
        elif rsi > 70:  # Overbought - sell signal
            self.signal(symbol, event.payload.timestamp, 'sell')
```

#### Multi-Asset Strategies
```python
class PairsTradingStrategy(BaseStrategy):
    def __init__(self, event_queue, data_handler, symbol1='AAPL', symbol2='MSFT'):
        super().__init__(event_queue, data_handler)
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.prices = defaultdict(list)
        self.spread_history = []
    
    def on_stock_event(self, event: StockEvent):
        symbol = event.payload.symbol
        price = event.payload.close
        
        self.prices[symbol].append(price)
        
        # Only trade when we have data for both symbols
        if (len(self.prices[self.symbol1]) > 0 and 
            len(self.prices[self.symbol2]) > 0):
            
            # Calculate spread
            price1 = self.prices[self.symbol1][-1]
            price2 = self.prices[self.symbol2][-1]
            spread = price1 - price2
            
            self.spread_history.append(spread)
            
            if len(self.spread_history) > 20:
                mean_spread = np.mean(self.spread_history[-20:])
                std_spread = np.std(self.spread_history[-20:])
                
                # Trade when spread deviates significantly
                if spread > mean_spread + 2 * std_spread:
                    # Spread too high - sell symbol1, buy symbol2
                    self.signal(self.symbol1, event.payload.timestamp, 'sell')
                    self.signal(self.symbol2, event.payload.timestamp, 'buy')
                elif spread < mean_spread - 2 * std_spread:
                    # Spread too low - buy symbol1, sell symbol2
                    self.signal(self.symbol1, event.payload.timestamp, 'buy')
                    self.signal(self.symbol2, event.payload.timestamp, 'sell')
```

---

## Data Preparation

### Required Data Format

Your data must be pandas DataFrames with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Price timestamp (timezone-aware preferred) |
| `symbol` | string | Stock symbol (e.g., 'AAPL', 'GOOGL') |
| `open` | float | Opening price |
| `high` | float | Highest price |
| `low` | float | Lowest price |
| `close` | float | Closing price |
| `volume` | int | Trading volume (optional) |

### Data Preparation Example

```python
import pandas as pd
import numpy as np

def create_sample_data(symbol, start_date, end_date, initial_price=100):
    """Create realistic sample OHLCV data"""
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate realistic price movements
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = initial_price * (1 + returns).cumprod()
    
    # Create OHLCV data with realistic intraday variation
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Previous close for gap calculation
        prev_close = prices[i-1] if i > 0 else close
        
        # Generate realistic OHLC
        gap = np.random.normal(0, 0.005)  # Overnight gap
        open_price = prev_close * (1 + gap)
        
        high_mult = 1 + abs(np.random.normal(0, 0.01))
        low_mult = 1 - abs(np.random.normal(0, 0.01))
        
        high = max(open_price, close) * high_mult
        low = min(open_price, close) * low_mult
        
        volume = int(np.random.normal(1000000, 200000))
        
        data.append({
            'timestamp': date,
            'symbol': symbol,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': max(volume, 100000)  # Minimum volume
        })
    
    return pd.DataFrame(data)

# Create multi-asset dataset
aapl_data = create_sample_data('AAPL', '2023-01-01', '2023-12-31', 150)
googl_data = create_sample_data('GOOGL', '2023-01-01', '2023-12-31', 2800)

data_dict = {
    'AAPL': aapl_data,
    'GOOGL': googl_data
}
```

### Loading Real Data

```python
# From CSV files
def load_csv_data(file_path, symbol):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['symbol'] = symbol
    return df

# From database
def load_from_database(symbol, start_date, end_date):
    # Example using your database connection
    from enode_quant.api.candles import get_stock_candles
    
    df = get_stock_candles(
        symbol=symbol,
        resolution='D',
        start_date=start_date,
        end_date=end_date,
        as_dataframe=True
    )
    return df

# Load multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT']
data_dict = {}

for symbol in symbols:
    data_dict[symbol] = load_from_database(symbol, '2023-01-01', '2023-12-31')
```

---

## Running Backtests

### Basic Backtest Execution

```python
from backtester import run_backtest, FixedSizeSizer

results = run_backtest(
    data_dict=data_dict,
    strategy_class=MovingAverageCrossStrategy,
    initial_cash=100_000.0,
    sizer=FixedSizeSizer(100)
)
```

### Advanced Configuration

```python
from backtester import run_backtest
from backtester.sizer import FixedSizeSizer
from backtester.risk import RiskManager, MaxPositionSizeRule, MaxCashUsageRule

# Custom commission model
def custom_commission(quantity, fill_price):
    \"\"\"$5 minimum, $0.01 per share\"\"\"
    return max(5.0, quantity * 0.01)

# Custom slippage model
def custom_slippage(order, market_price):
    \"\"\"0.05% slippage\"\"\"
    slippage_pct = 0.0005
    if order.signal == 'buy':
        return market_price * (1 + slippage_pct)
    else:
        return market_price * (1 - slippage_pct)

# Risk management
risk_manager = RiskManager([
    MaxPositionSizeRule(max_position_pct=0.20),  # Max 20% per position
    MaxCashUsageRule(reserve_cash=5000.0)        # Keep $5k reserve
])

# Run advanced backtest
results = run_backtest(
    data_dict=data_dict,
    strategy_class=MovingAverageCrossStrategy,
    initial_cash=500_000.0,
    sizer=FixedSizeSizer(200),
    risk_manager=risk_manager,
    commission_model=custom_commission,
    slippage_model=custom_slippage
)
```

### Accessing Results

```python
# Basic metrics
print(f"Total Return: {results.metrics.total_return:.2%}")
print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.metrics.max_drawdown:.2%}")

# Portfolio details
print(f"Final Cash: ${results.portfolio.current_cash:,.2f}")
print(f"Total Trades: {len(results.portfolio.trade_log)}")

# Equity curve
equity_curve = results.portfolio.equity_curve
final_equity = equity_curve[-1][1]
print(f"Final Equity: ${final_equity:,.2f}")

# Trade analysis
from backtester import get_trade_log_df
trades_df = get_trade_log_df(results.portfolio)
print(trades_df.head())
```

---

## Risk Management

The framework provides a comprehensive risk management system to protect your capital and enforce trading discipline.

### Available Risk Rules

#### 1. Position Size Limits
```python
from backtester.risk import MaxPositionSizeRule

# Limit any single position to 15% of portfolio
position_rule = MaxPositionSizeRule(max_position_pct=0.15)
```

#### 2. Cash Management
```python
from backtester.risk import MaxCashUsageRule

# Keep $10,000 in reserve
cash_rule = MaxCashUsageRule(reserve_cash=10000.0)
```

#### 3. Position Count Limits
```python
from backtester.risk import MaxPositionCountRule

# Maximum 8 simultaneous positions
count_rule = MaxPositionCountRule(max_positions=8)
```

#### 4. Drawdown Protection
```python
from backtester.risk import MaxDrawdownRule

# Stop trading if drawdown exceeds 15%
drawdown_rule = MaxDrawdownRule(max_drawdown_pct=0.15)
```

#### 5. Stop Loss
```python
from backtester.risk import StopLossRule

# 10% stop loss on all positions
stop_loss_rule = StopLossRule(stop_loss_pct=0.10)
```

### Combining Risk Rules

```python
from backtester.risk import RiskManager

# Create comprehensive risk management
risk_manager = RiskManager([
    MaxPositionSizeRule(max_position_pct=0.20),
    MaxCashUsageRule(reserve_cash=5000.0),
    MaxPositionCountRule(max_positions=10),
    MaxDrawdownRule(max_drawdown_pct=0.20),
    StopLossRule(stop_loss_pct=0.08)
])

# Use in backtest
results = run_backtest(
    data_dict=data_dict,
    strategy_class=MyStrategy,
    initial_cash=100_000.0,
    risk_manager=risk_manager
)
```

### Custom Risk Rules

```python
from backtester.risk import BaseRiskRule, RiskCheckResult

class VolatilityRule(BaseRiskRule):
    \"\"\"Reduce position size in high volatility periods\"\"\"
    
    def __init__(self, volatility_threshold=0.03):
        self.volatility_threshold = volatility_threshold
        self.price_history = {}
    
    def check(self, portfolio, signal_event, proposed_quantity, data_handler):
        symbol = signal_event.symbol
        current_price = data_handler.get_latest_bar_value(symbol, 'close')
        
        # Track price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(current_price)
        
        # Calculate volatility
        if len(self.price_history[symbol]) < 20:
            return RiskCheckResult(approved=True)
        
        prices = np.array(self.price_history[symbol][-20:])
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        if volatility > self.volatility_threshold:
            # Reduce position size by 50% in high volatility
            reduced_quantity = proposed_quantity // 2
            return RiskCheckResult(
                approved=True,
                modified_quantity=reduced_quantity,
                reason=f\"High volatility ({volatility:.3f}) - reduced position size\"
            )
        
        return RiskCheckResult(approved=True)
```

---

## Position Sizing

Position sizing determines how many shares to buy or sell for each signal. The framework provides flexible sizing strategies.

### Fixed Size Sizer

```python
from backtester.sizer import FixedSizeSizer

# Always trade 100 shares
sizer = FixedSizeSizer(100)
```

### Custom Sizers

```python
from backtester.sizer import BaseSizer

class PercentageOfEquitySizer(BaseSizer):
    \"\"\"Size positions as percentage of total equity\"\"\"
    
    def __init__(self, percentage=0.05):
        self.percentage = percentage  # 5% of equity per trade
    
    def size_order(self, portfolio, signal_event):
        # Calculate total equity
        total_holdings = sum(portfolio.current_holdings_value.values())
        total_equity = portfolio.current_cash + total_holdings
        
        # Get current price
        current_price = portfolio.data_handler.get_latest_bar_value(
            signal_event.symbol, 'close'
        )
        
        if current_price is None:
            return 0
        
        # Calculate position value and shares
        position_value = total_equity * self.percentage
        shares = int(position_value / current_price)
        
        return max(shares, 0)

class VolatilityAdjustedSizer(BaseSizer):
    \"\"\"Adjust position size based on volatility\"\"\"
    
    def __init__(self, base_size=100, lookback=20):
        self.base_size = base_size
        self.lookback = lookback
        self.price_history = {}
    
    def size_order(self, portfolio, signal_event):
        symbol = signal_event.symbol
        current_price = portfolio.data_handler.get_latest_bar_value(symbol, 'close')
        
        # Track price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(current_price)
        
        # Calculate volatility adjustment
        if len(self.price_history[symbol]) < self.lookback:
            return self.base_size
        
        prices = np.array(self.price_history[symbol][-self.lookback:])
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Reduce size for high volatility stocks
        volatility_adjustment = 1 / (1 + volatility * 10)
        adjusted_size = int(self.base_size * volatility_adjustment)
        
        return max(adjusted_size, 10)  # Minimum 10 shares
```

---

## Performance Analysis

The framework provides comprehensive performance analytics with statistical rigor.

### Key Performance Metrics

#### Return Metrics
- **Total Return**: Overall portfolio return from start to finish
- **Annualized Return**: Compound annual growth rate (CAGR)
- **Volatility**: Annualized standard deviation of returns

#### Risk-Adjusted Metrics
- **Sharpe Ratio**: $\\frac{\\text{Excess Return}}{\\text{Volatility}}$ - Risk-adjusted return
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
- **Calmar Ratio**: $\\frac{\\text{Annual Return}}{\\text{Max Drawdown}}$ - Return per unit of drawdown risk

#### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Expected loss in worst 5% of cases
- **Conditional VaR (CVaR)**: Average loss when VaR threshold is exceeded

#### Trade Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: $\\frac{\\text{Total Gains}}{\\text{Total Losses}}$ - Must be > 1.0 for profitability
- **Average Win/Loss**: Average profit per winning/losing trade

### Accessing Metrics

```python
# Run backtest
results = run_backtest(data_dict, MyStrategy, 100_000.0)

# Access all metrics
metrics = results.metrics

print(f\"Return Metrics:\")
print(f\"  Total Return: {metrics.total_return:.2%}\")
print(f\"  Annualized Return: {metrics.annualized_return:.2%}\")
print(f\"  Volatility: {metrics.volatility:.2%}\")

print(f\"\\nRisk-Adjusted Metrics:\")
print(f\"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}\")
print(f\"  Sortino Ratio: {metrics.sortino_ratio:.2f}\")
print(f\"  Calmar Ratio: {metrics.calmar_ratio:.2f}\")

print(f\"\\nRisk Metrics:\")
print(f\"  Max Drawdown: {metrics.max_drawdown:.2%}\")
print(f\"  VaR (95%): {metrics.var_95:.2%}\")
print(f\"  CVaR (95%): {metrics.cvar_95:.2%}\")

print(f\"\\nTrade Statistics:\")
print(f\"  Win Rate: {metrics.win_rate:.1%}\")
print(f\"  Profit Factor: {metrics.profit_factor:.2f}\")
print(f\"  Avg Win: ${metrics.avg_win:.2f}\")
print(f\"  Avg Loss: ${metrics.avg_loss:.2f}\")
```

### Statistical Analysis

#### Confidence Intervals
```python
# Bootstrap confidence intervals
return_ci = metrics.return_confidence_interval
sharpe_ci = metrics.sharpe_confidence_interval

print(f\"Return 95% CI: [{return_ci[0]:.3f}, {return_ci[1]:.3f}]\")
print(f\"Sharpe 95% CI: [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]\")
```

#### Monte Carlo Analysis
```python
# Access Monte Carlo simulation results
monte_carlo = results.monte_carlo

print(f\"Monte Carlo Scenarios:\")
percentiles = monte_carlo['final_value_percentiles']
print(f\"  Best Case (95%): ${percentiles['95%']:,.0f}\")
print(f\"  Expected (50%): ${percentiles['50%']:,.0f}\")
print(f\"  Worst Case (5%): ${percentiles['5%']:,.0f}\")

print(f\"\\nRisk Assessment:\")
print(f\"  Probability of Loss: {monte_carlo['probability_of_loss']:.1%}\")
```

### Trade Analysis

```python
from backtester import get_trade_log_df

# Get detailed trade log
trades_df = get_trade_log_df(results.portfolio)

print(\"Trade Summary:\")
print(f\"  Total Trades: {len(trades_df)}\")
print(f\"  Symbols Traded: {trades_df['symbol'].nunique()}\")

# Analyze by symbol
symbol_stats = trades_df.groupby('symbol').agg({
    'quantity': 'sum',
    'fill_price': 'mean',
    'commission': 'sum'
}).round(2)

print(\"\\nBy Symbol:\")
print(symbol_stats)

# Recent trades
print(\"\\nRecent Trades:\")
print(trades_df.tail())
```

---

## Interactive Dashboard

The framework includes a professional interactive dashboard for visual analysis and exploration of your backtest results.

### Launching the Dashboard

#### Method 1: Direct from Results (Recommended)
```python
# Run backtest and launch dashboard immediately
results = run_backtest(data_dict, MyStrategy, 100_000.0)
results.dashboard()  # Opens at http://localhost:8050
```

#### Method 2: From Saved Results
```python
# Save results first
results.save(\"my_backtest.json\")

# Launch later from CLI
# uv run python -m backtester.cli dashboard my_backtest.json
```

#### Method 3: Custom Port
```python
results.dashboard(port=8051)  # Custom port
results.dashboard(save=True)  # Save results file automatically
```

### Dashboard Features

#### 1. Performance Overview Tab
- **Equity Curve**: Interactive chart showing portfolio value over time
- **Buy & Hold Benchmark**: Comparison against equal-weight passive strategy
- **Key Metrics**: Summary cards with essential performance indicators
- **Portfolio Composition**: Shows which assets were traded

#### 2. Risk Analysis Tab
- **Returns Distribution**: Histogram of daily returns with statistical overlays
- **Drawdown Analysis**: Peak-to-trough decline visualization
- **Risk Metrics Table**: VaR, CVaR, volatility statistics
- **Statistical Explanations**: Educational content for each metric

#### 3. Trade Analysis Tab
- **Trade Statistics**: Win rate, profit factor, average win/loss
- **Recent Trades Table**: Detailed view of individual transactions
- **Trade Explanations**: Guidance on interpreting trade metrics

#### 4. Monte Carlo Tab
- **Scenario Analysis**: Range of possible future outcomes
- **Risk Assessment**: Probability of loss and confidence intervals
- **Simulation Explanations**: Understanding Monte Carlo methodology

### Understanding Dashboard Metrics

#### Performance Metrics Explained

**Total Return**
- Overall portfolio return from start to finish
- Formula: $\\frac{\\text{Final Value} - \\text{Initial Value}}{\\text{Initial Value}}$
- Good: > 10% annually, Excellent: > 20% annually

**Sharpe Ratio**
- Risk-adjusted return measure
- Formula: $\\frac{\\text{Mean Excess Return}}{\\text{Standard Deviation of Excess Returns}}$
- Interpretation: > 1.0 is good, > 2.0 is excellent, > 3.0 is exceptional

**Sortino Ratio**
- Like Sharpe but only penalizes downside volatility
- Formula: $\\frac{\\text{Mean Excess Return}}{\\text{Downside Deviation}}$
- Generally higher than Sharpe; > 1.5 is good

**Maximum Drawdown**
- Largest peak-to-trough decline
- Formula: $\\max_t \\left( \\frac{\\text{Peak}_t - \\text{Trough}_t}{\\text{Peak}_t} \\right)$
- Lower is better: < 10% is excellent, < 20% is acceptable

**Calmar Ratio**
- Annual return divided by maximum drawdown
- Formula: $\\frac{\\text{Annualized Return}}{|\\text{Max Drawdown}|}$
- Higher is better: > 1.0 is good, > 2.0 is excellent

#### Risk Metrics Explained

**Value at Risk (VaR)**
- Expected loss in worst 5% of cases
- 95% VaR of 3% means: \"95% of the time, daily loss won't exceed 3%\"
- Lower absolute values are better

**Conditional VaR (CVaR)**
- Average loss when VaR threshold is exceeded
- Also called \"Expected Shortfall\"
- Tells you how bad losses can be in the tail

**Volatility**
- Annualized standard deviation of returns
- Measures price fluctuation intensity
- Lower is generally better for risk-averse investors

#### Trade Metrics Explained

**Win Rate**
- Percentage of trades that were profitable
- Formula: $\\frac{\\text{Winning Trades}}{\\text{Total Trades}}$
- 50%+ is generally good, but depends on profit factor

**Profit Factor**
- Total gains divided by total losses
- Formula: $\\frac{\\sum \\text{Winning Trades}}{|\\sum \\text{Losing Trades}|}$
- Must be > 1.0 for profitability; > 1.5 is good

**Average Win vs Average Loss**
- Compare typical gain to typical loss
- Good strategies often have Average Win > Average Loss
- Can compensate for lower win rate

#### Monte Carlo Analysis Explained

**Scenario Percentiles**
- 95th percentile: Only 5% of simulations performed better
- 50th percentile: Median outcome (half above, half below)
- 5th percentile: Only 5% of simulations performed worse

**Probability of Loss**
- Likelihood of losing money based on historical patterns
- < 20%: Low risk strategy
- 20-40%: Moderate risk
- > 40%: High risk strategy

---

## Command Line Interface

The framework provides a powerful CLI for advanced users and automation.

### Available Commands

#### Launch Dashboard
```bash
# Launch dashboard from results file
uv run python -m backtester.cli dashboard results.json

# Custom port
uv run python -m backtester.cli dashboard results.json --port 8051

# Debug mode
uv run python -m backtester.cli dashboard results.json --debug
```

#### Analyze Results
```bash
# Display summary metrics
uv run python -m backtester.cli analyze results.json

# Display specific metric
uv run python -m backtester.cli analyze results.json --metric sharpe_ratio
```

#### Export Data
```bash
# Export to CSV
uv run python -m backtester.cli export results.json --format csv

# Custom output directory
uv run python -m backtester.cli export results.json --output ./exports/
```

#### Compare Results
```bash
# Compare two backtests
uv run python -m backtester.cli compare strategy1.json strategy2.json
```

### Automation Examples

#### Batch Analysis
```bash
#!/bin/bash
# Analyze multiple strategy results

for file in results/*.json; do
    echo \"Analyzing $file\"
    uv run python -m backtester.cli analyze \"$file\"
    echo \"---\"
done
```

#### Automated Reporting
```python
import subprocess
import json

def generate_report(results_file):
    \"\"\"Generate automated performance report\"\"\"
    
    # Get metrics via CLI
    result = subprocess.run([
        'uv', 'run', 'python', '-m', 'backtester.cli', 
        'analyze', results_file
    ], capture_output=True, text=True)
    
    print(f\"Report for {results_file}:\")
    print(result.stdout)
    
    # Export data
    subprocess.run([
        'uv', 'run', 'python', '-m', 'backtester.cli',
        'export', results_file, '--format', 'csv'
    ])

# Generate reports for all results
import glob
for results_file in glob.glob('*.json'):
    generate_report(results_file)
```

---

## Advanced Features

### Custom Execution Models

#### Advanced Commission Models
```python
def tiered_commission(quantity, fill_price):
    \"\"\"Tiered commission structure\"\"\"
    trade_value = quantity * fill_price
    
    if trade_value < 1000:
        return 9.95  # Flat fee for small trades
    elif trade_value < 10000:
        return trade_value * 0.005  # 0.5% for medium trades
    else:
        return trade_value * 0.003  # 0.3% for large trades

def per_share_commission(quantity, fill_price):
    \"\"\"Per-share commission with minimum\"\"\"
    return max(1.0, quantity * 0.005)  # $0.005 per share, $1 minimum
```

#### Realistic Slippage Models
```python
def market_impact_slippage(order, market_price):
    \"\"\"Slippage based on order size and volatility\"\"\"
    base_slippage = 0.0005  # 0.05% base slippage
    
    # Increase slippage for larger orders
    size_impact = min(order.quantity / 10000, 0.002)  # Max 0.2% size impact
    
    total_slippage = base_slippage + size_impact
    
    if order.signal == 'buy':
        return market_price * (1 + total_slippage)
    else:
        return market_price * (1 - total_slippage)

def time_based_slippage(order, market_price):
    \"\"\"Higher slippage during market open/close\"\"\"
    import datetime
    
    hour = order.timestamp.hour
    
    # Higher slippage during first/last hour
    if hour in [9, 15]:  # Market open/close hours
        slippage = 0.001  # 0.1%
    else:
        slippage = 0.0005  # 0.05%
    
    if order.signal == 'buy':
        return market_price * (1 + slippage)
    else:
        return market_price * (1 - slippage)
```

### Strategy Optimization

#### Parameter Optimization
```python
import itertools
from backtester import run_backtest

def optimize_strategy_parameters():
    \"\"\"Optimize moving average strategy parameters\"\"\"
    
    # Parameter ranges to test
    short_windows = [5, 10, 15, 20]
    long_windows = [20, 30, 40, 50]
    
    best_sharpe = -999
    best_params = None
    results_log = []
    
    for short, long in itertools.product(short_windows, long_windows):
        if short >= long:  # Skip invalid combinations
            continue
        
        # Create strategy class with parameters
        class OptimizedStrategy(MovingAverageCrossStrategy):
            def __init__(self, event_queue, data_handler):
                super().__init__(event_queue, data_handler, short, long)
        
        # Run backtest
        results = run_backtest(
            data_dict=data_dict,
            strategy_class=OptimizedStrategy,
            initial_cash=100_000.0
        )
        
        sharpe = results.metrics.sharpe_ratio
        results_log.append({
            'short_window': short,
            'long_window': long,
            'sharpe_ratio': sharpe,
            'total_return': results.metrics.total_return,
            'max_drawdown': results.metrics.max_drawdown
        })
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (short, long)
        
        print(f\"MA({short},{long}): Sharpe={sharpe:.2f}\")
    
    print(f\"\\nBest parameters: MA({best_params[0]},{best_params[1]}) with Sharpe={best_sharpe:.2f}\")
    
    return results_log, best_params

# Run optimization
results_log, best_params = optimize_strategy_parameters()
```

#### Walk-Forward Analysis
```python
def walk_forward_analysis(data_dict, strategy_class, window_months=6):
    \"\"\"Perform walk-forward analysis\"\"\"
    
    results = []
    
    # Get date range
    all_dates = []
    for df in data_dict.values():
        all_dates.extend(df['timestamp'].tolist())
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    
    current_date = start_date
    
    while current_date < end_date:
        # Define training and testing periods
        train_end = current_date + pd.DateOffset(months=window_months)
        test_end = train_end + pd.DateOffset(months=1)
        
        if test_end > end_date:
            break
        
        # Filter data for training period
        train_data = {}
        test_data = {}
        
        for symbol, df in data_dict.items():
            train_mask = (df['timestamp'] >= current_date) & (df['timestamp'] < train_end)
            test_mask = (df['timestamp'] >= train_end) & (df['timestamp'] < test_end)
            
            train_data[symbol] = df[train_mask].copy()
            test_data[symbol] = df[test_mask].copy()
        
        # Run backtest on test period
        if all(len(df) > 0 for df in test_data.values()):
            test_results = run_backtest(
                data_dict=test_data,
                strategy_class=strategy_class,
                initial_cash=100_000.0
            )
            
            results.append({
                'period_start': train_end,
                'period_end': test_end,
                'return': test_results.metrics.total_return,
                'sharpe': test_results.metrics.sharpe_ratio,
                'max_drawdown': test_results.metrics.max_drawdown
            })
        
        current_date = train_end
    
    return pd.DataFrame(results)

# Perform walk-forward analysis
wf_results = walk_forward_analysis(data_dict, MovingAverageCrossStrategy)
print(\"Walk-Forward Results:\")
print(wf_results)
```

### Multi-Strategy Portfolios

```python
class MultiStrategyPortfolio:
    \"\"\"Combine multiple strategies with allocation weights\"\"\"
    
    def __init__(self, strategies_config):
        self.strategies_config = strategies_config
        self.results = {}
    
    def run_all_strategies(self, data_dict, initial_cash):
        \"\"\"Run all strategies and combine results\"\"\"
        
        total_allocation = sum(config['weight'] for config in self.strategies_config)
        
        for name, config in self.strategies_config.items():
            # Allocate cash proportionally
            strategy_cash = initial_cash * (config['weight'] / total_allocation)
            
            # Run individual strategy
            results = run_backtest(
                data_dict=data_dict,
                strategy_class=config['strategy_class'],
                initial_cash=strategy_cash,
                sizer=config.get('sizer', FixedSizeSizer(50))
            )
            
            self.results[name] = results
        
        return self.combine_results()
    
    def combine_results(self):
        \"\"\"Combine individual strategy results\"\"\"
        
        combined_equity = []
        all_trades = []
        
        # Combine equity curves
        for name, results in self.results.items():
            weight = self.strategies_config[name]['weight']
            
            for timestamp, equity in results.portfolio.equity_curve:
                # Find or create combined equity point
                existing = next((item for item in combined_equity if item[0] == timestamp), None)
                
                if existing:
                    existing[1] += equity * weight
                else:
                    combined_equity.append([timestamp, equity * weight])
            
            # Combine trades
            all_trades.extend(results.portfolio.trade_log)
        
        # Sort by timestamp
        combined_equity.sort(key=lambda x: x[0])
        
        return {
            'combined_equity': combined_equity,
            'all_trades': all_trades,
            'individual_results': self.results
        }

# Example usage
strategies_config = {
    'momentum': {
        'strategy_class': MovingAverageCrossStrategy,
        'weight': 0.6,
        'sizer': FixedSizeSizer(100)
    },
    'mean_reversion': {
        'strategy_class': RSIStrategy,
        'weight': 0.4,
        'sizer': FixedSizeSizer(75)
    }
}

multi_portfolio = MultiStrategyPortfolio(strategies_config)
combined_results = multi_portfolio.run_all_strategies(data_dict, 200_000.0)
```

---

## Best Practices

### Strategy Development

1. **Start Simple**: Begin with basic strategies and add complexity gradually
2. **Avoid Overfitting**: Test on out-of-sample data to validate robustness
3. **Consider Transaction Costs**: Include realistic commissions and slippage
4. **Use Proper Risk Management**: Always implement position sizing and risk rules
5. **Validate Assumptions**: Ensure your strategy logic is sound and implementable

### Data Quality

1. **Clean Data**: Remove or handle missing values, outliers, and corporate actions
2. **Survivorship Bias**: Include delisted stocks in historical analysis
3. **Point-in-Time Data**: Use data as it was available at the time
4. **Consistent Frequency**: Ensure all symbols have the same data frequency
5. **Timezone Handling**: Use consistent timezone-aware timestamps

### Performance Analysis

1. **Multiple Metrics**: Don't rely on a single performance measure
2. **Risk-Adjusted Returns**: Always consider risk in performance evaluation
3. **Statistical Significance**: Use confidence intervals and statistical tests
4. **Benchmark Comparison**: Compare against relevant benchmarks
5. **Regime Analysis**: Test performance across different market conditions

### Risk Management

1. **Position Sizing**: Never risk more than you can afford to lose
2. **Diversification**: Don't put all capital in one position or strategy
3. **Stop Losses**: Implement systematic loss-cutting mechanisms
4. **Drawdown Limits**: Set maximum acceptable drawdown levels
5. **Regular Monitoring**: Continuously monitor strategy performance

### Production Considerations

1. **Latency**: Consider execution delays in live trading
2. **Market Impact**: Account for your trades affecting market prices
3. **Capacity**: Understand strategy capacity limitations
4. **Regime Changes**: Monitor for changes in market structure
5. **Technology Risk**: Have backup systems and error handling

### Common Pitfalls to Avoid

1. **Look-Ahead Bias**: Using future information in historical analysis
2. **Data Snooping**: Over-optimizing on the same dataset
3. **Ignoring Costs**: Underestimating transaction costs and slippage
4. **Curve Fitting**: Creating overly complex strategies that don't generalize
5. **Insufficient Testing**: Not testing across different market conditions

---

## Conclusion

The Enode Backtester provides a comprehensive framework for developing, testing, and analyzing quantitative trading strategies. Its event-driven architecture ensures realistic simulation while the modular design allows for extensive customization.

Key takeaways:
- Start with simple strategies and build complexity gradually
- Always include realistic transaction costs and risk management
- Use the interactive dashboard for deep performance analysis
- Validate strategies on out-of-sample data before deployment
- Consider the statistical significance of your results

For additional support and advanced features, refer to the API documentation and example strategies in the repository.

---

*Happy backtesting! 🚀*
