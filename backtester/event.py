from dataclasses import dataclass
import datetime
from .models import StockCandle

class Event:
    pass


class SecurityEvent(Event):
    pass

@dataclass
class StockEvent(SecurityEvent):
    """
    Stock candle for change in time
    Payload: Pydantic model StockCandle
    """
    # CORRECTED: Removed the "= payload"
    payload: StockCandle


@dataclass
class SignalEvent(Event):
    """
    We use this for deteriming the signal sent by a strategy
    Sent by: Strategy
    Read by: Portfolio
    """
    symbol: str
    signal: str #buy, sell, sell_all
    timestamp: datetime

@dataclass
class OrderEvent(Event):
    """
    We use this for deteriming the order sent by a strategy
    Sent by: Portfolio
    Read by: Execution handler
    """
    symbol: str
    signal: str #buy, sell
    timestamp: datetime
    quantity: int

# CORRECTED: Added @dataclass
@dataclass
class FillEvent(Event):
    """
    Carries the 'confirmation' of a filled order from the Broker.
    Sent by: ExecutionHandler
    Read by: Portfolio
    """
    symbol: str
    timestamp: datetime
    quantity: int
    fill_price: float
    commission: float
    signal: str #buy, sell


