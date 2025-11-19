"""Data models for the backtesting framework.

This module defines Pydantic models that represent financial data structures
used throughout the backtesting system. These models provide data validation,
serialization, and type safety.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class StockCandle(BaseModel):
    """OHLCV (Open, High, Low, Close, Volume) candle data for a stock.
    
    Represents market data for a specific stock at a specific timestamp.
    This is the core data structure that flows through the backtesting system
    as the payload of StockEvent objects.
    
    Attributes:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        stock_id: Optional database ID for the stock
        timestamp: When this candle data represents (e.g., daily close, hourly bar)
        resolution: Time resolution of the candle (e.g., '1D', '1H', '5m')
        open: Opening price for the time period
        high: Highest price during the time period
        low: Lowest price during the time period
        close: Closing price for the time period (most commonly used in strategies)
        volume: Number of shares traded during the time period
    
    Example:
        candle = StockCandle(
            symbol='AAPL',
            timestamp=datetime(2024, 1, 1),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000
        )
    
    Note:
        The model uses Pydantic for automatic validation and type conversion.
        It can be created from dictionaries (e.g., pandas DataFrame rows) using
        StockCandle.model_validate(dict_data).
    """
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    symbol: str
    stock_id: Optional[int] = None
    timestamp: datetime
    resolution: Optional[str] = None
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None