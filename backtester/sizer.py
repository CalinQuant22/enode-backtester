"""Position sizing module for determining trade quantities.

This module provides abstractions for position sizing logic - determining how many
shares to buy or sell when a trading signal is generated. Different sizing strategies
can implement risk management, portfolio allocation, and capital efficiency rules.
"""

import abc


class BaseSizer(metaclass=abc.ABCMeta):
    """Abstract base class for position sizing strategies.
    
    Sizers are responsible for determining the quantity of shares to trade
    when the Portfolio receives a SignalEvent from a Strategy. This is a
    critical component for risk management and capital allocation.
    
    Different sizing strategies might consider:
    - Fixed share quantities
    - Percentage of portfolio value
    - Risk-based sizing (volatility, stop-loss distance)
    - Kelly criterion optimization
    - Maximum position limits
    """

    @abc.abstractmethod
    def size_order(self, portfolio, signal_event) -> int:
        """Determine the number of shares to trade for a given signal.
        
        Args:
            portfolio: Portfolio object containing current cash, positions,
                      and holdings information
            signal_event: SignalEvent containing the trading signal details
                         (symbol, signal type, timestamp)
        
        Returns:
            int: Number of shares to trade. Should be positive for both
                 buy and sell signals. Return 0 to skip the trade.
        
        Note:
            The sizer should not determine trade direction (buy/sell) -
            that's specified in the signal_event. The sizer only determines
            quantity based on available capital, risk parameters, etc.
        """
        raise NotImplementedError()


class FixedSizeSizer(BaseSizer):
    """Simple sizer that always trades a fixed number of shares.
    
    This is the most basic sizing strategy, useful for:
    - Initial strategy development and testing
    - Strategies where position size doesn't matter for signal quality
    - Equal-weight portfolio approaches
    
    Attributes:
        default_size: Fixed number of shares to trade for every signal
    
    Example:
        sizer = FixedSizeSizer(100)  # Always trade 100 shares
        portfolio = Portfolio(..., sizer=sizer)
    
    Note:
        This sizer doesn't consider available cash, so it may generate
        orders that exceed available capital. The Portfolio should handle
        such cases by rejecting or reducing the order size.
    """

    def __init__(self, default_size: int = 50):
        """Initialize the fixed size sizer.
        
        Args:
            default_size: Number of shares to trade for every signal.
                         Must be positive.
        """
        self.default_size = default_size

    def size_order(self, portfolio, signal_event) -> int:
        """Return the fixed number of shares for every trade.
        
        Args:
            portfolio: Portfolio object (not used in this simple implementation)
            signal_event: SignalEvent (not used in this simple implementation)
        
        Returns:
            int: The fixed default_size specified during initialization
        
        Note:
            Future implementations might add cash availability checks:
            - For buy signals: ensure sufficient cash for the trade
            - For sell signals: ensure sufficient shares to sell
        """
        return self.default_size


