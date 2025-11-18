import abc
import queue
from collections.abc import Iterator
from typing import Any

import pandas as pd

from .event import StockEvent
from .models import StockCandle


class BaseDataHandler(abc.ABC):
    @abc.abstractmethod
    def update_bars(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_latest_bar_value(self, symbol: str, field: str) -> Any:
        raise NotImplementedError()


class DataFrameDataHandler(BaseDataHandler):
    def __init__(self, event_queue: queue.Queue, data_dict: dict[str, pd.DataFrame]):
        self.event_queue = event_queue
        self._iterators: dict[str, Iterator[dict[str, Any]]] = {}
        self._latest_bars: dict[str, StockCandle] = {}

        for symbol, dataframe in data_dict.items():
            if dataframe.empty:
                continue

            df_sorted = dataframe.sort_values(by="timestamp", ascending=True).copy()
            df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], utc=True)

            records = df_sorted.to_dict(orient="records")
            self._iterators[symbol] = iter(records)

    def update_bars(self) -> None:
        data_available = False

        for symbol, iterator in self._iterators.items():
            try:
                bar_data = next(iterator)
            except StopIteration:
                continue

            candle = StockCandle.model_validate(bar_data)
            self._latest_bars[symbol] = candle
            self.event_queue.put(StockEvent(payload=candle))
            data_available = True

        if not data_available:
            raise StopIteration

    def get_latest_bar_value(self, symbol: str, field: str) -> Any:
        candle = self._latest_bars.get(symbol)
        if candle is None:
            return None
        return getattr(candle, field, None)
