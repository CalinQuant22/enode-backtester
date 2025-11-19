"""Order execution simulation module.

This module simulates the execution of trading orders with realistic market conditions
including slippage (price impact) and commission costs. It converts OrderEvents into
FillEvents with actual execution details.
"""

import abc
import queue
from datetime import datetime

from .event import OrderEvent, FillEvent 
from .data import BaseDataHandler


def default_slippage_model(order: OrderEvent, latest_price: float) -> float:
    """Default slippage model that applies small price impact.
    
    Simulates the realistic scenario where:
    - Buy orders execute slightly above market price (paying the spread)
    - Sell orders execute slightly below market price (paying the spread)
    
    Args:
        order: OrderEvent containing order details
        latest_price: Current market price from data handler
    
    Returns:
        float: Adjusted execution price including slippage
    
    Note:
        This simple model applies 0.01% slippage. Real-world slippage
        depends on order size, market volatility, liquidity, and timing.
    """
    if order.signal == 'buy':
        return latest_price * 1.0001  # Pay 0.01% more (unfavorable)
    elif order.signal == 'sell':
        return latest_price * 0.9999  # Receive 0.01% less (unfavorable)
    return latest_price


def default_commission_model(quantity: int, fill_price: float) -> float:
    """Default commission model with per-share pricing.
    
    Args:
        quantity: Number of shares traded
        fill_price: Execution price per share (not used in this model)
    
    Returns:
        float: Total commission cost for the trade
    
    Note:
        Uses $0.005 per share, which is typical for discount brokers.
        Real commission structures might include:
        - Flat fees per trade
        - Percentage of trade value
        - Tiered pricing based on volume
        - Different rates for different asset classes
    """
    return quantity * 0.005


class BaseExecutionHandler(metaclass=abc.ABCMeta):
    """Abstract base class for order execution handlers.
    
    Execution handlers simulate the process of sending orders to a broker
    and receiving fill confirmations. They model real-world trading costs
    and market conditions that affect execution quality.
    
    Different implementations might simulate:
    - Market orders vs. limit orders
    - Different broker commission structures
    - Market impact based on order size
    - Partial fills and order rejection
    - Live trading integration
    """

    @abc.abstractmethod
    def on_order_event(self, event: OrderEvent) -> None:
        """Process an order and generate a fill event.
        
        Args:
            event: OrderEvent containing order details (symbol, quantity, etc.)
        
        Note:
            Implementations should:
            1. Validate the order (check market data availability)
            2. Determine execution price (apply slippage model)
            3. Calculate transaction costs (commission, fees)
            4. Create and emit a FillEvent with execution details
        """
        raise NotImplementedError()


class SimulatedExecutionHandler(BaseExecutionHandler):
    """Simulated execution handler for backtesting.
    
    This handler simulates order execution by:
    1. Taking orders from the Portfolio
    2. Applying configurable slippage and commission models
    3. Generating immediate fills (assumes infinite liquidity)
    4. Emitting FillEvents back to the Portfolio
    
    The simulation assumes:
    - Orders are filled immediately at current market price + slippage
    - Infinite market liquidity (no partial fills)
    - No order rejection (all valid orders execute)
    - Market orders only (no limit order logic)
    
    Attributes:
        event_queue: Queue for emitting FillEvent objects
        data_handler: Source of current market prices
        slippage_model: Function to calculate price impact
        commission_model: Function to calculate transaction costs
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        data_handler: BaseDataHandler,
        commission_model=default_commission_model,
        slippage_model=default_slippage_model,
    ):
        """Initialize the simulated execution handler.
        
        Args:
            event_queue: Central event queue for the backtesting system
            data_handler: Interface to current market data
            commission_model: Function(quantity, fill_price) -> commission_cost
            slippage_model: Function(order, market_price) -> execution_price
        """
        self.event_queue = event_queue
        self.data_handler = data_handler
        self.slippage_model = slippage_model
        self.commission_model = commission_model

    def on_order_event(self, event: OrderEvent) -> None:
        """Execute an order and generate a fill confirmation.
        
        This method simulates the complete order execution process:
        1. Validates the order type and market data availability
        2. Retrieves current market price for the symbol
        3. Applies slippage model to determine execution price
        4. Calculates commission costs
        5. Creates and emits a FillEvent with execution details
        
        Args:
            event: OrderEvent containing order details
        
        Note:
            If market data is not available for the symbol, the order
            is skipped with a warning. This can happen if:
            - The symbol has no more historical data
            - There's a data quality issue
            - The symbol was delisted
        """
        # Validate order type
        if event.signal not in ['buy', 'sell']:
            return

        # Get current market price
        latest_price = self.data_handler.get_latest_bar_value(event.symbol, 'close')
        
        if latest_price is None:
            print(f"Warning: No market data for {event.symbol} on {event.timestamp}. Order skipped.")
            return

        # Apply slippage to get realistic execution price
        fill_price = self.slippage_model(event, latest_price)
        
        # Calculate transaction costs
        commission = self.commission_model(event.quantity, fill_price)

        # Create fill confirmation
        fill_event = FillEvent(
            symbol=event.symbol,
            timestamp=event.timestamp,
            quantity=event.quantity,
            fill_price=fill_price,
            commission=commission,
            signal=event.signal
        )

        # Emit fill event for Portfolio to process
        self.event_queue.put(fill_event)

    