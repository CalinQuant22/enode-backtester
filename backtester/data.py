"""Data handling module for streaming historical market data.

This module provides abstractions for feeding historical market data into the
backtesting engine. It converts static datasets into a time-ordered stream
of market events, ensuring proper chronological processing.
"""

import abc
import queue
from collections.abc import Iterator
from typing import Any

import pandas as pd

from .event import StockEvent
from .models import StockCandle


class BaseDataHandler(abc.ABC):
    """Abstract base class for data handlers.
    
    Data handlers are responsible for:
    1. Managing historical market data
    2. Streaming data chronologically into the event system
    3. Providing access to current market data for other components
    
    This abstraction allows for different data sources (CSV files, databases,
    live feeds) while maintaining a consistent interface.
    """

    @abc.abstractmethod
    def update_bars(self) -> None:
        """Advance to the next time step and emit market data events.
        
        This method should:
        1. Move to the next chronological data point for all symbols
        2. Create StockEvent objects for new data
        3. Put events into the event queue
        4. Raise StopIteration when no more data is available
        
        Raises:
            StopIteration: When all historical data has been processed
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_latest_bar_value(self, symbol: str, field: str) -> Any:
        """Get the most recent value for a specific field of a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            field: Field name (e.g., 'close', 'volume', 'high')
        
        Returns:
            The requested field value, or None if not available
        """
        raise NotImplementedError()


class DataFrameDataHandler(BaseDataHandler):
    """Data handler that streams pandas DataFrames chronologically.
    
    Takes a dictionary of pandas DataFrames (one per symbol) and converts them
    into a time-ordered stream of StockEvent objects. This is the primary way
    to feed historical data into the backtesting system.
    
    The handler ensures proper chronological ordering across all symbols,
    preventing look-ahead bias by only revealing data as time progresses.
    
    Attributes:
        event_queue: Queue for emitting StockEvent objects
        _iterators: Per-symbol iterators over historical data
        _latest_bars: Cache of most recent data for each symbol
    """

    def __init__(self, event_queue: queue.Queue, data_dict: dict[str, pd.DataFrame]):
        """Initialize the data handler with historical market data.
        
        Args:
            event_queue: Central event queue for the backtesting system
            data_dict: Dictionary mapping symbol names to pandas DataFrames.
                      Each DataFrame must have columns: symbol, timestamp, 
                      open, high, low, close, volume (volume optional)
        
        Note:
            DataFrames are automatically sorted by timestamp to ensure
            chronological processing. Empty DataFrames are ignored.
        """
        self.event_queue = event_queue
        self._iterators: dict[str, Iterator[dict[str, Any]]] = {}
        self._latest_bars: dict[str, StockCandle] = {}

        for symbol, dataframe in data_dict.items():
            if dataframe.empty:
                continue

            # Ensure chronological ordering and proper datetime format
            df_sorted = dataframe.sort_values(by="timestamp", ascending=True).copy()
            df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], utc=True)

            # Convert to iterator over dictionary records
            records = df_sorted.to_dict(orient="records")
            self._iterators[symbol] = iter(records)

    def update_bars(self) -> None:
        """Advance one time step and emit StockEvents for all symbols.
        
        This method processes the next chronological data point for each symbol,
        creates StockCandle objects, and emits them as StockEvents. This ensures
        that strategies receive market data in proper time order.
        
        The method advances ALL symbols simultaneously to maintain temporal
        consistency across the portfolio.
        
        Raises:
            StopIteration: When all symbols have exhausted their data,
                          signaling the end of the backtest
        
        Note:
            If a symbol runs out of data before others, it's skipped in
            subsequent time steps. The backtest continues until ALL symbols
            are exhausted.
        """
        data_available = False

        for symbol, iterator in self._iterators.items():
            try:
                bar_data = next(iterator)
            except StopIteration:
                # This symbol has no more data, skip it
                continue

            # Convert raw data to validated StockCandle model
            candle = StockCandle.model_validate(bar_data)
            self._latest_bars[symbol] = candle
            
            # Emit market data event
            self.event_queue.put(StockEvent(payload=candle))
            data_available = True

        if not data_available:
            # All symbols exhausted, end the backtest
            raise StopIteration

    def get_latest_bar_value(self, symbol: str, field: str) -> Any:
        """Retrieve the most recent value for a specific field of a symbol.
        
        This method provides access to the current market data, which is used by:
        1. Portfolio for position valuation
        2. ExecutionHandler for determining fill prices
        3. Strategies for accessing current market conditions
        
        Args:
            symbol: Stock symbol to query (e.g., 'AAPL')
            field: Field name from StockCandle (e.g., 'close', 'volume', 'high')
        
        Returns:
            The requested field value, or None if the symbol hasn't been
            processed yet or the field doesn't exist
        
        Example:
            current_price = data_handler.get_latest_bar_value('AAPL', 'close')
            current_volume = data_handler.get_latest_bar_value('AAPL', 'volume')
        """
        candle = self._latest_bars.get(symbol)
        if candle is None:
            return None
        return getattr(candle, field, None)
