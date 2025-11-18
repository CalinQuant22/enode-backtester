# my_strategies.py
from collections import defaultdict
import numpy as np
from backtester.strategy import BaseStrategy
from backtester.event import StockEvent

class MovingAverageCrossStrategy(BaseStrategy):
    """
    A simple strategy that buys when the Short SMA crosses above the Long SMA.
    """
    def __init__(self, event_queue, data_handler, short_window=10, long_window=30):
        super().__init__(event_queue, data_handler)
        self.short_window = short_window
        self.long_window = long_window
        
        # Store price history for each symbol: {'AAPL': [150, 151, ...]}
        self.prices = defaultdict(list)
        # Track our current stance to avoid repeating signals: {'AAPL': 'FLAT'}
        self.market_position = defaultdict(lambda: 'FLAT')

    def on_stock_event(self, event: StockEvent):
        symbol = event.payload.symbol
        price = event.payload.close
        timestamp = event.payload.timestamp

        # 1. Store the new price
        self.prices[symbol].append(price)

        # 2. Wait until we have enough data for the long window
        if len(self.prices[symbol]) < self.long_window:
            return

        # 3. Calculate SMAs (using the last N prices)
        # Note: In production, using deque or optimized rolling calc is faster
        price_history = self.prices[symbol]
        
        short_sma = np.mean(price_history[-self.short_window:])
        long_sma = np.mean(price_history[-self.long_window:])

        # 4. Generate Signals
        current_pos = self.market_position[symbol]

        # BUY SIGNAL: Short SMA crosses above Long SMA
        if short_sma > long_sma and current_pos != 'LONG':
            self.signal(symbol, timestamp, 'buy')
            self.market_position[symbol] = 'LONG'
            print(f"[{timestamp}] BUY {symbol} @ {price:.2f} (SMA{self.short_window}: {short_sma:.2f}, SMA{self.long_window}: {long_sma:.2f})")

        # SELL SIGNAL: Short SMA crosses below Long SMA
        elif short_sma < long_sma and current_pos != 'FLAT': # Assuming we just sell to close
            self.signal(symbol, timestamp, 'sell')
            self.market_position[symbol] = 'FLAT'
            print(f"[{timestamp}] SELL {symbol} @ {price:.2f}")