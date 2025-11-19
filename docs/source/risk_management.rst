Risk Management
===============

Risk management is crucial for protecting capital and ensuring long-term trading success. The framework provides a comprehensive risk management system with built-in rules and the ability to create custom risk controls.

Overview
--------

The risk management system validates every order before execution, ensuring trades comply with your risk parameters. Risk rules can:

- Limit position sizes
- Enforce cash reserves
- Implement stop losses
- Control maximum drawdown
- Limit number of simultaneous positions

Available Risk Rules
--------------------

Position Size Limits
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from backtester.risk import MaxPositionSizeRule

    # Limit any single position to 15% of portfolio
    position_rule = MaxPositionSizeRule(max_position_pct=0.15)

This rule prevents over-concentration in any single asset, ensuring diversification.

Cash Management
~~~~~~~~~~~~~~~

.. code-block:: python

    from backtester.risk import MaxCashUsageRule

    # Keep $10,000 in reserve
    cash_rule = MaxCashUsageRule(reserve_cash=10000.0)

Maintains a cash buffer for opportunities and unexpected expenses.

Position Count Limits
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from backtester.risk import MaxPositionCountRule

    # Maximum 8 simultaneous positions
    count_rule = MaxPositionCountRule(max_positions=8)

Prevents over-diversification and helps maintain focus on best opportunities.

Drawdown Protection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from backtester.risk import MaxDrawdownRule

    # Stop trading if drawdown exceeds 15%
    drawdown_rule = MaxDrawdownRule(max_drawdown_pct=0.15)

Automatically halts trading when losses exceed acceptable levels.

Stop Loss Protection
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from backtester.risk import StopLossRule

    # 10% stop loss on all positions
    stop_loss_rule = StopLossRule(stop_loss_pct=0.10)

Automatically exits positions when they decline beyond acceptable levels.

Combining Risk Rules
--------------------

Create comprehensive risk management by combining multiple rules:

.. code-block:: python

    from backtester.risk import RiskManager

    # Create comprehensive risk management
    risk_manager = RiskManager([
        MaxPositionSizeRule(max_position_pct=0.20),
        MaxCashUsageRule(reserve_cash=5000.0),
        MaxPositionCountRule(max_positions=10),
        MaxDrawdownRule(max_drawdown_pct=0.20),
        StopLossRule(stop_loss_pct=0.08)
    ])

    # Use in backtest
    results = run_backtest(
        data_dict=data_dict,
        strategy_class=MyStrategy,
        initial_cash=100_000.0,
        risk_manager=risk_manager
    )

Custom Risk Rules
-----------------

Create custom risk rules by inheriting from ``BaseRiskRule``:

.. code-block:: python

    from backtester.risk import BaseRiskRule, RiskCheckResult
    import numpy as np

    class VolatilityRule(BaseRiskRule):
        """Reduce position size in high volatility periods"""
        
        def __init__(self, volatility_threshold=0.03):
            self.volatility_threshold = volatility_threshold
            self.price_history = {}
        
        def check(self, portfolio, signal_event, proposed_quantity, data_handler):
            symbol = signal_event.symbol
            current_price = data_handler.get_latest_bar_value(symbol, 'close')
            
            # Track price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(current_price)
            
            # Calculate volatility
            if len(self.price_history[symbol]) < 20:
                return RiskCheckResult(approved=True)
            
            prices = np.array(self.price_history[symbol][-20:])
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            if volatility > self.volatility_threshold:
                # Reduce position size by 50% in high volatility
                reduced_quantity = proposed_quantity // 2
                return RiskCheckResult(
                    approved=True,
                    modified_quantity=reduced_quantity,
                    reason=f"High volatility ({volatility:.3f}) - reduced position size"
                )
            
            return RiskCheckResult(approved=True)

Advanced Risk Rules
-------------------

Correlation-Based Risk
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class CorrelationRule(BaseRiskRule):
        """Limit exposure to highly correlated assets"""
        
        def __init__(self, max_correlation=0.8, lookback_days=30):
            self.max_correlation = max_correlation
            self.lookback_days = lookback_days
            self.price_history = defaultdict(list)
        
        def check(self, portfolio, signal_event, proposed_quantity, data_handler):
            symbol = signal_event.symbol
            
            # Update price history
            current_price = data_handler.get_latest_bar_value(symbol, 'close')
            self.price_history[symbol].append(current_price)
            
            # Check correlation with existing positions
            for existing_symbol in portfolio.current_positions:
                if existing_symbol == symbol:
                    continue
                
                if len(self.price_history[symbol]) >= self.lookback_days and \
                   len(self.price_history[existing_symbol]) >= self.lookback_days:
                    
                    # Calculate correlation
                    returns1 = np.diff(self.price_history[symbol][-self.lookback_days:])
                    returns2 = np.diff(self.price_history[existing_symbol][-self.lookback_days:])
                    
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    
                    if abs(correlation) > self.max_correlation:
                        return RiskCheckResult(
                            approved=False,
                            reason=f"High correlation ({correlation:.2f}) with {existing_symbol}"
                        )
            
            return RiskCheckResult(approved=True)

Sector Concentration Rule
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class SectorConcentrationRule(BaseRiskRule):
        """Limit exposure to any single sector"""
        
        def __init__(self, sector_map, max_sector_pct=0.30):
            self.sector_map = sector_map  # {symbol: sector}
            self.max_sector_pct = max_sector_pct
        
        def check(self, portfolio, signal_event, proposed_quantity, data_handler):
            symbol = signal_event.symbol
            
            if symbol not in self.sector_map:
                return RiskCheckResult(approved=True)
            
            target_sector = self.sector_map[symbol]
            current_price = data_handler.get_latest_bar_value(symbol, 'close')
            proposed_value = proposed_quantity * current_price
            
            # Calculate current sector exposure
            total_equity = portfolio.current_cash + sum(portfolio.current_holdings_value.values())
            sector_exposure = 0
            
            for pos_symbol, position in portfolio.current_positions.items():
                if self.sector_map.get(pos_symbol) == target_sector:
                    sector_exposure += portfolio.current_holdings_value[pos_symbol]
            
            # Check if adding this position would exceed sector limit
            new_sector_exposure = sector_exposure + proposed_value
            sector_pct = new_sector_exposure / total_equity
            
            if sector_pct > self.max_sector_pct:
                # Calculate maximum allowed quantity
                max_sector_value = total_equity * self.max_sector_pct
                max_additional_value = max_sector_value - sector_exposure
                max_quantity = int(max_additional_value / current_price)
                
                if max_quantity <= 0:
                    return RiskCheckResult(
                        approved=False,
                        reason=f"Sector {target_sector} exposure would exceed {self.max_sector_pct:.1%}"
                    )
                else:
                    return RiskCheckResult(
                        approved=True,
                        modified_quantity=max_quantity,
                        reason=f"Reduced quantity to maintain sector limit"
                    )
            
            return RiskCheckResult(approved=True)

Position Sizing Integration
---------------------------

Risk management works closely with position sizing. The framework applies risk rules after the sizer determines the initial position size:

.. code-block:: python

    # Flow: Strategy Signal → Sizer → Risk Manager → Order

    class RiskAwareSizer(BaseSizer):
        """Sizer that considers risk metrics"""
        
        def __init__(self, base_percentage=0.05, max_risk_per_trade=0.02):
            self.base_percentage = base_percentage
            self.max_risk_per_trade = max_risk_per_trade
        
        def size_order(self, portfolio, signal_event):
            # Calculate base position size
            total_equity = portfolio.current_cash + sum(portfolio.current_holdings_value.values())
            base_position_value = total_equity * self.base_percentage
            
            # Get current price
            current_price = portfolio.data_handler.get_latest_bar_value(
                signal_event.symbol, 'close'
            )
            
            if current_price is None:
                return 0
            
            # Calculate position size based on risk
            # Assume 5% stop loss for risk calculation
            stop_loss_pct = 0.05
            risk_per_share = current_price * stop_loss_pct
            
            # Maximum shares based on risk limit
            max_risk_value = total_equity * self.max_risk_per_trade
            max_shares_by_risk = int(max_risk_value / risk_per_share)
            
            # Maximum shares based on position size
            max_shares_by_size = int(base_position_value / current_price)
            
            # Take the smaller of the two
            return min(max_shares_by_risk, max_shares_by_size)

Risk Monitoring
---------------

Monitor risk metrics during backtesting:

.. code-block:: python

    class RiskMonitor:
        """Monitor risk metrics during backtest"""
        
        def __init__(self):
            self.risk_events = []
            self.drawdown_history = []
        
        def update(self, portfolio, timestamp):
            # Calculate current drawdown
            if portfolio.equity_curve:
                current_equity = portfolio.equity_curve[-1][1]
                peak_equity = max(eq[1] for eq in portfolio.equity_curve)
                drawdown = (peak_equity - current_equity) / peak_equity
                
                self.drawdown_history.append((timestamp, drawdown))
                
                # Log significant risk events
                if drawdown > 0.10:  # 10% drawdown
                    self.risk_events.append({
                        'timestamp': timestamp,
                        'type': 'high_drawdown',
                        'value': drawdown,
                        'description': f'Drawdown reached {drawdown:.2%}'
                    })
        
        def get_risk_summary(self):
            if not self.drawdown_history:
                return {}
            
            drawdowns = [dd[1] for dd in self.drawdown_history]
            
            return {
                'max_drawdown': max(drawdowns),
                'avg_drawdown': np.mean(drawdowns),
                'drawdown_periods': len([dd for dd in drawdowns if dd > 0.05]),
                'risk_events': len(self.risk_events)
            }

Risk-Adjusted Performance
-------------------------

Evaluate strategies using risk-adjusted metrics:

.. code-block:: python

    def calculate_risk_metrics(portfolio):
        """Calculate comprehensive risk metrics"""
        
        if not portfolio.equity_curve:
            return {}
        
        # Extract returns
        equity_values = [eq[1] for eq in portfolio.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        # Basic metrics
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Drawdown analysis
        peak = equity_values[0]
        drawdowns = []
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        max_drawdown = max(drawdowns)
        
        # Risk-adjusted returns
        if volatility > 0:
            sharpe_ratio = (total_return * 252) / volatility  # Simplified
        else:
            sharpe_ratio = 0
        
        if max_drawdown > 0:
            calmar_ratio = (total_return * 252) / max_drawdown
        else:
            calmar_ratio = float('inf')
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'avg_drawdown': np.mean(drawdowns)
        }

Best Practices
--------------

Risk Rule Design
~~~~~~~~~~~~~~~~

1. **Start Conservative**: Begin with strict risk limits and relax gradually
2. **Test Thoroughly**: Validate risk rules across different market conditions
3. **Monitor Performance**: Track how risk rules affect strategy performance
4. **Regular Review**: Periodically review and adjust risk parameters
5. **Document Rationale**: Keep clear documentation of why each rule exists

Implementation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Layer Defense**: Use multiple complementary risk rules
2. **Real-time Monitoring**: Implement alerts for risk threshold breaches
3. **Graceful Degradation**: Ensure system continues operating when rules trigger
4. **Performance Impact**: Consider computational cost of complex risk rules
5. **Backtesting Validation**: Test risk rules in historical scenarios

Common Risk Scenarios
~~~~~~~~~~~~~~~~~~~~~

**Market Crash Protection**
  - Maximum drawdown limits
  - Volatility-based position sizing
  - Correlation monitoring

**Concentration Risk**
  - Position size limits
  - Sector exposure limits
  - Single-name concentration

**Liquidity Risk**
  - Volume-based position sizing
  - Market impact considerations
  - Cash reserve requirements

**Operational Risk**
  - System failure contingencies
  - Data quality checks
  - Execution monitoring

Next Steps
----------

- Implement risk rules in your strategies
- Use the :doc:`dashboard` to monitor risk metrics
- Explore :doc:`examples` for risk management patterns