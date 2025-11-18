# backtester/strategy.py
import abc
import queue
from datetime import datetime

# Import YOUR event class names
from .event import StockEvent, SignalEvent 
from .data import BaseDataHandler


class BaseStrategy(metaclass=abc.ABCMeta):

    def __init__(self, event_queue: queue.Queue, data_handler: BaseDataHandler):
        self.event_queue = event_queue
        self.data_handler = data_handler

    @abc.abstractmethod
    def on_stock_event(self, event: StockEvent):
        raise NotImplementedError()

    def signal(self, symbol: str, timestamp: datetime, signal: str):
        signal_event = SignalEvent(
            symbol = symbol,
            timestamp = timestamp,
            signal = signal
        )
        self.event_queue.put(signal_event)
 