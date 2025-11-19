#!/usr/bin/env python3
"""
Test script to verify dashboard functionality.
Run this to test the dashboard with sample data.
"""

import pandas as pd
import numpy as np
from backtester import run_backtest, FixedSizeSizer
from backtester.strategy import BaseStrategy
from backtester.event import StockEvent
from backtester.dashboard.app import launch_dashboard

class SampleStrategy(BaseStrategy):
    """Simple moving average crossover strategy for testing."""
    
    def __init__(self, event_queue, data_handler):
        super().__init__(event_queue, data_handler)
        self.prices = []
        self.short_window = 10
        self.long_window = 20
        
    def on_stock_event(self, event: StockEvent) -> None:
        price = event.payload.close
        self.prices.append(price)
        
        if len(self.prices) < self.long_window:
            return
            
        # Calculate moving averages
        short_ma = sum(self.prices[-self.short_window:]) / self.short_window
        long_ma = sum(self.prices[-self.long_window:]) / self.long_window
        
        # Generate signals
        if short_ma > long_ma and len(self.prices) % 3 == 0:  # Buy signal
            self.signal(event.payload.symbol, event.payload.timestamp, 'buy')
        elif short_ma < long_ma and len(self.prices) % 5 == 0:  # Sell signal
            self.signal(event.payload.symbol, event.payload.timestamp, 'sell')

def create_sample_data():
    """Create realistic sample market data."""
    
    # Generate 1 year of daily data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create realistic price movement with trend and volatility
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Slight upward bias
    prices = 100 * (1 + returns).cumprod()
    
    # Add some realistic intraday variation
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
    opens = prices * (1 + np.random.normal(0, 0.005, len(dates)))
    volumes = np.random.randint(100000, 1000000, len(dates))
    
    return pd.DataFrame({
        'timestamp': dates,
        'symbol': 'SAMPLE',
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })

def main():
    """Run the dashboard test."""
    
    print("🚀 Creating sample data...")
    sample_data = create_sample_data()
    
    print("📊 Running backtest...")
    portfolio = run_backtest(
        data_dict={'SAMPLE': sample_data},
        strategy_class=SampleStrategy,
        initial_cash=100000.0,
        sizer=FixedSizeSizer(default_size=100)
    )
    
    print(f"✅ Backtest complete!")
    print(f"   - Final equity: ${portfolio.equity_curve[-1][1]:,.2f}")
    print(f"   - Total trades: {len(portfolio.trade_log)}")
    
    print("🌐 Launching dashboard...")
    print("   Dashboard will open at http://localhost:8050")
    print("   Press Ctrl+C to stop the server")
    
    # Launch dashboard
    launch_dashboard(portfolio=portfolio, port=8050, debug=True)

if __name__ == "__main__":
    main()