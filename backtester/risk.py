"""Risk management module for portfolio protection and position control.

This module provides a flexible rule-based risk management system that:
- Validates orders before execution
- Enforces position size limits
- Implements stop-loss mechanisms
- Monitors portfolio-level risk metrics
- Prevents excessive drawdowns

Risk rules are composable and can be combined to create sophisticated
risk management strategies.
"""

import abc
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .portfolio import Portfolio
    from .event import SignalEvent
    from .data import BaseDataHandler


@dataclass
class RiskCheckResult:
    """Result of a risk check evaluation.
    
    Attributes:
        approved: Whether the order is approved
        modified_quantity: Adjusted quantity (None = use original)
        reason: Explanation for rejection or modification
    """
    approved: bool
    modified_quantity: Optional[int] = None
    reason: str = ""


class BaseRiskRule(abc.ABC):
    """Abstract base class for risk management rules.
    
    Risk rules evaluate proposed trades and can:
    - Approve the trade as-is
    - Modify the trade quantity
    - Reject the trade entirely
    
    Rules are evaluated in sequence by the RiskManager.
    """
    
    @abc.abstractmethod
    def check(
        self,
        portfolio: "Portfolio",
        signal_event: "SignalEvent",
        proposed_quantity: int,
        data_handler: "BaseDataHandler"
    ) -> RiskCheckResult:
        """Evaluate if a proposed trade meets risk criteria.
        
        Args:
            portfolio: Current portfolio state
            signal_event: Trading signal being evaluated
            proposed_quantity: Number of shares proposed by sizer
            data_handler: Access to current market data
            
        Returns:
            RiskCheckResult indicating approval, modification, or rejection
        """
        raise NotImplementedError()


class MaxPositionSizeRule(BaseRiskRule):
    """Limit position size as percentage of portfolio equity.
    
    Prevents over-concentration in a single position by capping
    the maximum value of any position relative to total equity.
    
    Example:
        rule = MaxPositionSizeRule(max_position_pct=0.20)  # Max 20% per position
    """
    
    def __init__(self, max_position_pct: float = 0.25):
        """Initialize position size limit rule.
        
        Args:
            max_position_pct: Maximum position size as fraction of equity (0.25 = 25%)
        """
        self.max_position_pct = max_position_pct
    
    def check(
        self,
        portfolio: "Portfolio",
        signal_event: "SignalEvent",
        proposed_quantity: int,
        data_handler: "BaseDataHandler"
    ) -> RiskCheckResult:
        """Check if position size would exceed limit."""
        if signal_event.signal != 'buy':
            return RiskCheckResult(approved=True)
        
        # Get current price
        current_price = data_handler.get_latest_bar_value(signal_event.symbol, 'close')
        if current_price is None:
            return RiskCheckResult(approved=False, reason="No price data available")
        
        # Calculate total equity
        total_holdings = sum(portfolio.current_holdings_value.values())
        total_equity = portfolio.current_cash + total_holdings
        
        # Calculate proposed position value
        current_position_value = portfolio.current_holdings_value.get(signal_event.symbol, 0)
        proposed_position_value = current_position_value + (proposed_quantity * current_price)
        
        # Check if within limit
        max_position_value = total_equity * self.max_position_pct
        
        if proposed_position_value <= max_position_value:
            return RiskCheckResult(approved=True)
        
        # Calculate reduced quantity
        available_value = max_position_value - current_position_value
        if available_value <= 0:
            return RiskCheckResult(
                approved=False,
                reason=f"Position already at max size ({self.max_position_pct:.1%} of equity)"
            )
        
        reduced_quantity = int(available_value / current_price)
        if reduced_quantity <= 0:
            return RiskCheckResult(
                approved=False,
                reason=f"Cannot add to position without exceeding {self.max_position_pct:.1%} limit"
            )
        
        return RiskCheckResult(
            approved=True,
            modified_quantity=reduced_quantity,
            reason=f"Reduced from {proposed_quantity} to {reduced_quantity} shares (position size limit)"
        )


class MaxCashUsageRule(BaseRiskRule):
    """Ensure sufficient cash is available for buy orders.
    
    Prevents orders that would result in negative cash balance,
    accounting for commission costs.
    """
    
    def __init__(self, reserve_cash: float = 0.0):
        """Initialize cash availability rule.
        
        Args:
            reserve_cash: Minimum cash to keep in reserve (emergency buffer)
        """
        self.reserve_cash = reserve_cash
    
    def check(
        self,
        portfolio: "Portfolio",
        signal_event: "SignalEvent",
        proposed_quantity: int,
        data_handler: "BaseDataHandler"
    ) -> RiskCheckResult:
        """Check if sufficient cash is available."""
        if signal_event.signal != 'buy':
            return RiskCheckResult(approved=True)
        
        current_price = data_handler.get_latest_bar_value(signal_event.symbol, 'close')
        if current_price is None:
            return RiskCheckResult(approved=False, reason="No price data available")
        
        # Estimate total cost (price + estimated commission)
        estimated_commission = proposed_quantity * 0.005  # Use default commission model
        total_cost = (proposed_quantity * current_price) + estimated_commission
        
        available_cash = portfolio.current_cash - self.reserve_cash
        
        if total_cost <= available_cash:
            return RiskCheckResult(approved=True)
        
        # Calculate affordable quantity
        affordable_quantity = int((available_cash - estimated_commission) / current_price)
        
        if affordable_quantity <= 0:
            return RiskCheckResult(
                approved=False,
                reason=f"Insufficient cash (need ${total_cost:.2f}, have ${available_cash:.2f})"
            )
        
        return RiskCheckResult(
            approved=True,
            modified_quantity=affordable_quantity,
            reason=f"Reduced from {proposed_quantity} to {affordable_quantity} shares (cash limit)"
        )


class MaxPositionCountRule(BaseRiskRule):
    """Limit the total number of open positions.
    
    Prevents over-diversification and helps maintain focus on
    best opportunities.
    """
    
    def __init__(self, max_positions: int = 10):
        """Initialize position count limit rule.
        
        Args:
            max_positions: Maximum number of simultaneous positions
        """
        self.max_positions = max_positions
    
    def check(
        self,
        portfolio: "Portfolio",
        signal_event: "SignalEvent",
        proposed_quantity: int,
        data_handler: "BaseDataHandler"
    ) -> RiskCheckResult:
        """Check if adding position would exceed limit."""
        if signal_event.signal != 'buy':
            return RiskCheckResult(approved=True)
        
        # Count current non-zero positions
        current_positions = sum(1 for qty in portfolio.current_positions.values() if qty != 0)
        
        # Check if we already have a position in this symbol
        has_position = portfolio.current_positions.get(signal_event.symbol, 0) != 0
        
        if has_position or current_positions < self.max_positions:
            return RiskCheckResult(approved=True)
        
        return RiskCheckResult(
            approved=False,
            reason=f"Maximum positions reached ({self.max_positions})"
        )


class MaxDrawdownRule(BaseRiskRule):
    """Stop trading if portfolio drawdown exceeds threshold.
    
    Protects capital by halting trading during severe drawdowns,
    preventing further losses.
    """
    
    def __init__(self, max_drawdown_pct: float = 0.20):
        """Initialize maximum drawdown rule.
        
        Args:
            max_drawdown_pct: Maximum allowed drawdown (0.20 = 20%)
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_equity = None
    
    def check(
        self,
        portfolio: "Portfolio",
        signal_event: "SignalEvent",
        proposed_quantity: int,
        data_handler: "BaseDataHandler"
    ) -> RiskCheckResult:
        """Check if current drawdown exceeds limit."""
        # Calculate current equity
        total_holdings = sum(portfolio.current_holdings_value.values())
        current_equity = portfolio.current_cash + total_holdings
        
        # Track peak equity
        if self.peak_equity is None or current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        # Calculate drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        if drawdown > self.max_drawdown_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Max drawdown exceeded ({drawdown:.1%} > {self.max_drawdown_pct:.1%})"
            )
        
        return RiskCheckResult(approved=True)


class StopLossRule(BaseRiskRule):
    """Automatic stop-loss for positions below entry price.
    
    Generates sell signals when positions fall below stop-loss threshold.
    Note: This rule generates signals rather than blocking orders.
    """
    
    def __init__(self, stop_loss_pct: float = 0.10):
        """Initialize stop-loss rule.
        
        Args:
            stop_loss_pct: Stop loss threshold (0.10 = 10% loss)
        """
        self.stop_loss_pct = stop_loss_pct
        self.entry_prices: dict[str, float] = {}
    
    def check(
        self,
        portfolio: "Portfolio",
        signal_event: "SignalEvent",
        proposed_quantity: int,
        data_handler: "BaseDataHandler"
    ) -> RiskCheckResult:
        """Check stop-loss conditions and track entry prices."""
        symbol = signal_event.symbol
        current_price = data_handler.get_latest_bar_value(symbol, 'close')
        
        if current_price is None:
            return RiskCheckResult(approved=True)
        
        # Track entry price on buys
        if signal_event.signal == 'buy':
            if symbol not in self.entry_prices:
                self.entry_prices[symbol] = current_price
            return RiskCheckResult(approved=True)
        
        # Clear entry price on sells
        if signal_event.signal == 'sell':
            self.entry_prices.pop(symbol, None)
            return RiskCheckResult(approved=True)
        
        return RiskCheckResult(approved=True)
    
    def check_stop_loss(
        self,
        portfolio: "Portfolio",
        data_handler: "BaseDataHandler"
    ) -> list[str]:
        """Check all positions for stop-loss triggers.
        
        Returns:
            List of symbols that hit stop-loss
        """
        triggered_symbols = []
        
        for symbol, entry_price in self.entry_prices.items():
            position = portfolio.current_positions.get(symbol, 0)
            if position <= 0:
                continue
            
            current_price = data_handler.get_latest_bar_value(symbol, 'close')
            if current_price is None:
                continue
            
            loss_pct = (entry_price - current_price) / entry_price
            
            if loss_pct >= self.stop_loss_pct:
                triggered_symbols.append(symbol)
        
        return triggered_symbols


class RiskManager:
    """Centralized risk management coordinator.
    
    Evaluates proposed trades against a set of risk rules and
    determines if orders should be approved, modified, or rejected.
    
    Rules are evaluated in sequence, and the most restrictive
    result is applied.
    """
    
    def __init__(self, rules: Optional[list[BaseRiskRule]] = None):
        """Initialize risk manager with rules.
        
        Args:
            rules: List of risk rules to apply (empty list = no restrictions)
        """
        self.rules = rules or []
    
    def add_rule(self, rule: BaseRiskRule) -> None:
        """Add a risk rule to the manager.
        
        Args:
            rule: Risk rule to add
        """
        self.rules.append(rule)
    
    def evaluate_order(
        self,
        portfolio: "Portfolio",
        signal_event: "SignalEvent",
        proposed_quantity: int,
        data_handler: "BaseDataHandler"
    ) -> RiskCheckResult:
        """Evaluate a proposed order against all risk rules.
        
        Args:
            portfolio: Current portfolio state
            signal_event: Trading signal
            proposed_quantity: Proposed order size
            data_handler: Market data access
            
        Returns:
            RiskCheckResult with final decision
        """
        if not self.rules:
            return RiskCheckResult(approved=True)
        
        current_quantity = proposed_quantity
        reasons = []
        
        for rule in self.rules:
            result = rule.check(portfolio, signal_event, current_quantity, data_handler)
            
            if not result.approved:
                return result  # Immediate rejection
            
            if result.modified_quantity is not None:
                current_quantity = result.modified_quantity
                if result.reason:
                    reasons.append(result.reason)
        
        # All rules passed
        if current_quantity != proposed_quantity:
            return RiskCheckResult(
                approved=True,
                modified_quantity=current_quantity,
                reason="; ".join(reasons)
            )
        
        return RiskCheckResult(approved=True)
