import queue
from datetime import datetime
from collections import defaultdict

from .event import StockEvent, SignalEvent, OrderEvent, FillEvent
from .data import BaseDataHandler
from .sizer import BaseSizer, FixedSizeSizer 


#We are gonna need to add more metrics 
class BasePortfolio(object):
    pass

class Portfolio(BasePortfolio):

    def __init__(
        self,
        event_queue: queue.Queue,
        data_handler: BaseDataHandler,
        initial_cash: float, 
        sizer: BaseSizer = FixedSizeSizer(50)
    ):
        self.event_queue = event_queue
        self.data_handler = data_handler
        self.sizer = sizer
        self.initial_cash = initial_cash

        self.current_cash = initial_cash
        self.current_positions = defaultdict(int)
        self.current_holdings_value = defaultdict(float)

        self.equity_curve = []
        self.trade_log = [] 
    def on_stock_event(self, event: StockEvent):
        symbol = event.payload.symbol 

        if self.current_positions[symbol] != 0: 
            latest_price = event.payload.close
            market_value = latest_price * self.current_positions[symbol]
            self.current_holdings_value[symbol] = market_value

        self.record_equity(event.payload.timestamp)

    
    def record_equity(self, timestamp: datetime):
        total_holdings = sum(self.current_holdings_value.values())
        total_equity = self.current_cash + total_holdings
        self.equity_curve.append((timestamp, total_equity))

    
    def on_signal_event(self, event: SignalEvent):

        quantity = self.sizer.size_order(self, event) 
        
        if quantity == 0:
            return

        if event.signal == 'buy' or event.signal == 'sell':
            order_event = OrderEvent(
                symbol = event.symbol,
                signal = event.signal,
                timestamp = event.timestamp,
                quantity = quantity
            )
            self.event_queue.put(order_event)

    
    def on_fill_event(self, event: FillEvent):
        
        if event.signal == 'buy':
            self.current_cash -= (event.fill_price * event.quantity) + event.commission
            self.current_positions[event.symbol] += event.quantity
        elif event.signal == 'sell':
            self.current_cash += (event.fill_price * event.quantity) - event.commission
            self.current_positions[event.symbol] -= event.quantity

        
        latest_price = self.data_handler.get_latest_bar_value(event.symbol, 'close')

        if latest_price is not None:
            market_value = self.current_positions[event.symbol] * latest_price
            self.current_holdings_value[event.symbol] = market_value
        
        self.trade_log.append(event)