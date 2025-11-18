import queue
from typing import Type

import pandas as pd

from .analysis import generate_full_tear_sheet, get_trade_log_df
from .data import DataFrameDataHandler
from .engine import BacktestEngine
from .execution import (
    SimulatedExecutionHandler,
    default_commission_model,
    default_slippage_model,
)
from .portfolio import Portfolio
from .sizer import BaseSizer, FixedSizeSizer
from .strategy import BaseStrategy


def run_backtest(
    data_dict: dict[str, pd.DataFrame],
    strategy_class: Type[BaseStrategy],
    initial_cash: float,
    sizer: BaseSizer | None = None,
    commission_model=None,
    slippage_model=None,
):
    print("--- Initializing Backtest Components ---")

    event_queue = queue.Queue()
    data_handler = DataFrameDataHandler(event_queue, data_dict)
    strategy = strategy_class(event_queue, data_handler)
    portfolio = Portfolio(
        event_queue,
        data_handler,
        initial_cash,
        sizer=sizer or FixedSizeSizer(50),
    )

    execution_handler = SimulatedExecutionHandler(
        event_queue,
        data_handler,
        commission_model=commission_model or default_commission_model,
        slippage_model=slippage_model or default_slippage_model,
    )
    engine = BacktestEngine(event_queue, data_handler, strategy, portfolio, execution_handler)

    final_portfolio = engine.run()

    print("--- Backtest Complete. Returning results. ---")
    return final_portfolio


__all__ = [
    "run_backtest",
    "generate_full_tear_sheet",
    "get_trade_log_df",
    "FixedSizeSizer",
]