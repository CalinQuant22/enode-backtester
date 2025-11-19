"""Strategy module for implementing trading logic.

This module provides the base strategy interface that all trading strategies must implement.
Strategies receive market data events, maintain internal state (indicators, price history),
and emit trading signals when conditions are met.
"""

import abc
import queue
from datetime import datetime

from .event import StockEvent, SignalEvent 
from .data import BaseDataHandler


class BaseStrategy(metaclass=abc.ABCMeta):
    """Abstract base class for all trading strategies.
    
    Strategies are event-driven components that:
    1. Receive StockEvent objects containing market data (OHLCV candles)
    2. Maintain internal state like price history, technical indicators, etc.
    3. Generate SignalEvent objects when trading conditions are met
    4. Cannot directly place orders - only signal trading intent
    
    The strategy operates in the event loop by reacting to incoming market data
    and emitting signals that the Portfolio will convert into actual orders.
    
    Attributes:
        event_queue: Central event queue for inter-component communication
        data_handler: Provides access to current and historical market data
    """

    def __init__(self, event_queue: queue.Queue, data_handler: BaseDataHandler):
        """Initialize the strategy with event queue and data handler.
        
        Args:
            event_queue: Central queue for passing events between components
            data_handler: Interface to access market data and latest bar values
        """
        self.event_queue = event_queue
        self.data_handler = data_handler

    @abc.abstractmethod
    def on_stock_event(self, event: StockEvent) -> None:
        """Process incoming market data and generate trading signals.
        
        This is the core method that must be implemented by all strategies.
        It receives new market data (price candles) and should:
        1. Update internal state (price history, indicators, etc.)
        2. Evaluate trading conditions
        3. Call self.signal() when buy/sell conditions are met
        
        Args:
            event: StockEvent containing a StockCandle with OHLCV data
                  for a specific symbol and timestamp
        
        Note:
            This method should never directly place orders. Use self.signal()
            to emit SignalEvents that the Portfolio will process.
        """
        raise NotImplementedError()

    def signal(self, symbol: str, timestamp: datetime, signal: str) -> None:
        """Emit a trading signal to the event queue.
        
        This is the primary way strategies communicate trading intent.
        The signal will be picked up by the Portfolio, which will determine
        position sizing and create actual orders.
        
        Args:
            symbol: Stock symbol to trade (e.g., 'AAPL', 'GOOGL')
            timestamp: When the signal was generated (should match current data timestamp)
            signal: Trading action - typically 'buy', 'sell', or 'sell_all'
        
        Example:
            # In your strategy's on_stock_event method:
            if price > moving_average:
                self.signal('AAPL', event.payload.timestamp, 'buy')
        """
        signal_event = SignalEvent(
            symbol=symbol,
            timestamp=timestamp,
            signal=signal
        )
        self.event_queue.put(signal_event)
 