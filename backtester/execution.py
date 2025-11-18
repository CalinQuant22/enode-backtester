import abc
import queue
from datetime import datetime

from .event import OrderEvent, FillEvent 
from .data import BaseDataHandler

#we have to come up with a way to model slippage/ commission i guess we get from broker
def default_slippage_model(order: OrderEvent, latest_price: float):
    if order.signal == 'buy':
        return latest_price * 1.0001 
    elif order.signal == 'sell':
        return latest_price * 0.9999
    return latest_price

def default_commission_model(quantity: int, fill_price: float):
    return quantity * 0.005


class BaseExecutionHandler(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def on_order_event(self, event: OrderEvent):
        raise NotImplementedError()


#we add order to the queue
class SimulatedExecutionHandler(BaseExecutionHandler):
    def __init__(
        self,
        event_queue: queue.Queue,
        data_handler: BaseDataHandler,
        commission_model=default_commission_model,
        slippage_model=default_slippage_model,
    ):
        self.event_queue = event_queue
        self.data_handler = data_handler
        self.slippage_model = slippage_model
        self.commission_model = commission_model

    def on_order_event(self, event: OrderEvent):
        if event.signal not in ['buy', 'sell']:
            return

        latest_price = self.data_handler.get_latest_bar_value(event.symbol, 'close')
        
        if latest_price is None:

            print(f"Warning: No market data for {event.symbol} on {event.timestamp}. Order skipped.")
            return

        fill_price = self.slippage_model(event, latest_price)
        
        commission = self.commission_model(event.quantity, fill_price)

        fill_event = FillEvent(
            symbol = event.symbol,
            timestamp = event.timestamp,
            quantity = event.quantity,
            fill_price = fill_price,
            commission = commission,
            signal = event.signal
        )

        self.event_queue.put(fill_event)

    