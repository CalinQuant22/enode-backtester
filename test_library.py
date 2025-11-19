#!/usr/bin/env python3
"""Quick test to verify the backtester library works end-to-end."""

import pandas as pd
from datetime import datetime, timedelta
from backtester import run_backtest, FixedSizeSizer, get_trade_log_df, RiskManager, MaxPositionSizeRule, MaxCashUsageRule
from backtester.strategy import BaseStrategy
from backtester.event import StockEvent

class SimpleStrategy(BaseStrategy):
    """Buy when price > 50, sell when price < 45."""
    
    def on_stock_event(self, event: StockEvent) -> None:
        price = event.payload.close
        if price > 50:
            self.signal(event.payload.symbol, event.payload.timestamp, "buy")
        elif price < 45:
            self.signal(event.payload.symbol, event.payload.timestamp, "sell")

# Create sample data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
prices = [40 + i * 0.5 + (i % 10) * 2 for i in range(100)]  # Trending up with noise

sample_data = pd.DataFrame({
    'symbol': 'TEST',
    'timestamp': dates,
    'open': prices,
    'high': [p + 1 for p in prices],
    'low': [p - 1 for p in prices], 
    'close': prices,
    'volume': [1000] * 100
})

data_dict = {'TEST': sample_data}

# Test without risk management
print("Running backtest WITHOUT risk management...")
portfolio_no_risk = run_backtest(
    data_dict=data_dict,
    strategy_class=SimpleStrategy,
    initial_cash=10000.0,
    sizer=FixedSizeSizer(100)  # Larger size to trigger risk limits
)

# Test with risk management
print("\nRunning backtest WITH risk management...")
risk_manager = RiskManager([
    MaxPositionSizeRule(max_position_pct=0.20),  # Max 20% per position
    MaxCashUsageRule(reserve_cash=1000.0)        # Keep $1000 in reserve
])

portfolio_with_risk = run_backtest(
    data_dict=data_dict,
    strategy_class=SimpleStrategy,
    initial_cash=10000.0,
    sizer=FixedSizeSizer(100),  # Same large size
    risk_manager=risk_manager
)

# Compare results
print("\n=== COMPARISON ===")
print(f"WITHOUT risk: Final cash: ${portfolio_no_risk.current_cash:.2f}, Equity: ${portfolio_no_risk.equity_curve[-1][1]:.2f}")
print(f"WITH risk:    Final cash: ${portfolio_with_risk.current_cash:.2f}, Equity: ${portfolio_with_risk.equity_curve[-1][1]:.2f}")

trade_log_no_risk = get_trade_log_df(portfolio_no_risk)
trade_log_with_risk = get_trade_log_df(portfolio_with_risk)
print(f"\nTrades WITHOUT risk: {len(trade_log_no_risk)}")
print(f"Trades WITH risk:    {len(trade_log_with_risk)}")

if len(trade_log_with_risk) > 0:
    print("\nRisk-managed trades (note smaller quantities):")
    print(trade_log_with_risk.head())

print("✅ Library test completed successfully!")