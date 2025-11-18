import queue
from datetime import datetime

from .event import Event, StockEvent, SignalEvent, OrderEvent, FillEvent 
from .data import BaseDataHandler
from .strategy import BaseStrategy
from .portfolio import BasePortfolio
from .execution import BaseExecutionHandler



class BacktestEngine:


    def __init__(
        self,
        event_queue: queue.Queue,
        data_handler: BaseDataHandler,
        strategy: BaseStrategy,
        portfolio: BasePortfolio,
        execution_handler: BaseExecutionHandler,

    ):
        self.event_queue = event_queue
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler

 

    def run(self):

        print(f"--- Backtest Started at {datetime.now()} ---")

        while True:
            try:
                self.data_handler.update_bars()
            except StopIteration:
                break

            while not self.event_queue.empty():
                try:
                    event = self.event_queue.get(block=False)

                except queue.Empty:
                    break
                #update candles 
                if isinstance(event, StockEvent):
                    self.strategy.on_stock_event(event)
                    self.portfolio.on_stock_event(event)

                elif isinstance(event, SignalEvent):
                    self.portfolio.on_signal_event(event)

                elif isinstance(event, OrderEvent):
                    self.execution_handler.on_order_event(event)

                elif isinstance(event, FillEvent):
                    self.portfolio.on_fill_event(event)

        print(f"--- Backtest Finished at {datetime.now()} ---")

        return self.portfolio





            






        

