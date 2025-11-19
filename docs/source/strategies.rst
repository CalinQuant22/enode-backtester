Strategy Development
===================

Creating effective trading strategies is the core of successful backtesting. This guide covers strategy development patterns, best practices, and advanced techniques.

Strategy Base Class
-------------------

All strategies must inherit from ``BaseStrategy`` and implement the ``on_stock_event`` method:

.. code-block:: python

    from backtester.strategy import BaseStrategy
    from backtester.event import StockEvent

    class MyStrategy(BaseStrategy):
        def __init__(self, event_queue, data_handler):
            super().__init__(event_queue, data_handler)
            # Initialize your strategy state here
            
        def on_stock_event(self, event: StockEvent) -> None:
            # Your trading logic here
            # Access price data: event.payload.close, event.payload.high, etc.
            # Generate signals: self.signal(symbol, timestamp, 'buy'/'sell')
            pass

Strategy Development Pattern
----------------------------

1. **Initialize State**: Set up indicators, price history, and parameters
2. **Process Market Data**: Update indicators and internal state
3. **Generate Signals**: Emit buy/sell signals when conditions are met
4. **Maintain Position Awareness**: Track your current market position

Example Strategies
------------------

Moving Average Crossover
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from collections import defaultdict
    import numpy as np
    from backtester.strategy import BaseStrategy
    from backtester.event import StockEvent

    class MovingAverageCrossStrategy(BaseStrategy):
        def __init__(self, event_queue, data_handler, short_window=10, long_window=30):
            super().__init__(event_queue, data_handler)
            self.short_window = short_window
            self.long_window = long_window
            
            # Store price history for each symbol
            self.prices = defaultdict(list)
            # Track current position to avoid duplicate signals
            self.position = defaultdict(lambda: 'FLAT')  # 'FLAT', 'LONG'

        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            timestamp = event.payload.timestamp

            # 1. Update price history
            self.prices[symbol].append(price)

            # 2. Wait for sufficient data
            if len(self.prices[symbol]) < self.long_window:
                return

            # 3. Calculate moving averages
            prices = self.prices[symbol]
            short_ma = np.mean(prices[-self.short_window:])
            long_ma = np.mean(prices[-self.long_window:])

            # 4. Generate signals
            current_pos = self.position[symbol]

            # Buy when short MA crosses above long MA
            if short_ma > long_ma and current_pos == 'FLAT':
                self.signal(symbol, timestamp, 'buy')
                self.position[symbol] = 'LONG'

            # Sell when short MA crosses below long MA
            elif short_ma < long_ma and current_pos == 'LONG':
                self.signal(symbol, timestamp, 'sell')
                self.position[symbol] = 'FLAT'

RSI Strategy
~~~~~~~~~~~~

.. code-block:: python

    class RSIStrategy(BaseStrategy):
        def __init__(self, event_queue, data_handler, rsi_period=14):
            super().__init__(event_queue, data_handler)
            self.rsi_period = rsi_period
            self.prices = defaultdict(list)
        
        def calculate_rsi(self, prices):
            """Calculate RSI indicator"""
            if len(prices) < self.rsi_period + 1:
                return None
                
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-self.rsi_period:])
            avg_loss = np.mean(losses[-self.rsi_period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            
            self.prices[symbol].append(price)
            
            rsi = self.calculate_rsi(self.prices[symbol])
            if rsi is None:
                return
                
            # RSI oversold/overbought strategy
            if rsi < 30:  # Oversold - buy signal
                self.signal(symbol, event.payload.timestamp, 'buy')
            elif rsi > 70:  # Overbought - sell signal
                self.signal(symbol, event.payload.timestamp, 'sell')

Pairs Trading Strategy
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class PairsTradingStrategy(BaseStrategy):
        def __init__(self, event_queue, data_handler, symbol1='AAPL', symbol2='MSFT'):
            super().__init__(event_queue, data_handler)
            self.symbol1 = symbol1
            self.symbol2 = symbol2
            self.prices = defaultdict(list)
            self.spread_history = []
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            
            self.prices[symbol].append(price)
            
            # Only trade when we have data for both symbols
            if (len(self.prices[self.symbol1]) > 0 and 
                len(self.prices[self.symbol2]) > 0):
                
                # Calculate spread
                price1 = self.prices[self.symbol1][-1]
                price2 = self.prices[self.symbol2][-1]
                spread = price1 - price2
                
                self.spread_history.append(spread)
                
                if len(self.spread_history) > 20:
                    mean_spread = np.mean(self.spread_history[-20:])
                    std_spread = np.std(self.spread_history[-20:])
                    
                    # Trade when spread deviates significantly
                    if spread > mean_spread + 2 * std_spread:
                        # Spread too high - sell symbol1, buy symbol2
                        self.signal(self.symbol1, event.payload.timestamp, 'sell')
                        self.signal(self.symbol2, event.payload.timestamp, 'buy')
                    elif spread < mean_spread - 2 * std_spread:
                        # Spread too low - buy symbol1, sell symbol2
                        self.signal(self.symbol1, event.payload.timestamp, 'buy')
                        self.signal(self.symbol2, event.payload.timestamp, 'sell')

Advanced Strategy Techniques
----------------------------

Technical Indicators
~~~~~~~~~~~~~~~~~~~~

Building a library of reusable technical indicators:

.. code-block:: python

    class TechnicalIndicators:
        @staticmethod
        def sma(prices, window):
            """Simple Moving Average"""
            if len(prices) < window:
                return None
            return np.mean(prices[-window:])
        
        @staticmethod
        def ema(prices, window, alpha=None):
            """Exponential Moving Average"""
            if alpha is None:
                alpha = 2.0 / (window + 1)
            
            if len(prices) < 2:
                return prices[-1] if prices else None
            
            ema_prev = prices[0]
            for price in prices[1:]:
                ema_prev = alpha * price + (1 - alpha) * ema_prev
            
            return ema_prev
        
        @staticmethod
        def bollinger_bands(prices, window=20, num_std=2):
            """Bollinger Bands"""
            if len(prices) < window:
                return None, None, None
            
            sma = np.mean(prices[-window:])
            std = np.std(prices[-window:])
            
            upper = sma + (num_std * std)
            lower = sma - (num_std * std)
            
            return upper, sma, lower

Multi-Timeframe Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class MultiTimeframeStrategy(BaseStrategy):
        def __init__(self, event_queue, data_handler):
            super().__init__(event_queue, data_handler)
            self.daily_prices = defaultdict(list)
            self.weekly_prices = defaultdict(list)
            self.last_weekly_update = {}
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            timestamp = event.payload.timestamp
            
            # Update daily prices
            self.daily_prices[symbol].append(price)
            
            # Update weekly prices (simplified - every 5 days)
            if symbol not in self.last_weekly_update:
                self.last_weekly_update[symbol] = timestamp
                self.weekly_prices[symbol].append(price)
            elif (timestamp - self.last_weekly_update[symbol]).days >= 5:
                self.weekly_prices[symbol].append(price)
                self.last_weekly_update[symbol] = timestamp
            
            # Strategy logic using both timeframes
            if len(self.daily_prices[symbol]) >= 20 and len(self.weekly_prices[symbol]) >= 4:
                # Daily trend
                daily_sma = np.mean(self.daily_prices[symbol][-10:])
                daily_trend = price > daily_sma
                
                # Weekly trend
                weekly_sma = np.mean(self.weekly_prices[symbol][-4:])
                weekly_trend = self.weekly_prices[symbol][-1] > weekly_sma
                
                # Only trade when both timeframes align
                if daily_trend and weekly_trend:
                    self.signal(symbol, timestamp, 'buy')
                elif not daily_trend and not weekly_trend:
                    self.signal(symbol, timestamp, 'sell')

Position Management
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class PositionManagedStrategy(BaseStrategy):
        def __init__(self, event_queue, data_handler):
            super().__init__(event_queue, data_handler)
            self.positions = defaultdict(dict)  # Track position details
            self.entry_prices = defaultdict(float)
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            timestamp = event.payload.timestamp
            
            # Your entry logic here
            if self.should_enter(symbol, price):
                self.signal(symbol, timestamp, 'buy')
                self.entry_prices[symbol] = price
                self.positions[symbol] = {
                    'entry_price': price,
                    'entry_time': timestamp,
                    'stop_loss': price * 0.95,  # 5% stop loss
                    'take_profit': price * 1.10  # 10% take profit
                }
            
            # Position management for existing positions
            elif symbol in self.positions:
                pos = self.positions[symbol]
                
                # Stop loss
                if price <= pos['stop_loss']:
                    self.signal(symbol, timestamp, 'sell')
                    del self.positions[symbol]
                    del self.entry_prices[symbol]
                
                # Take profit
                elif price >= pos['take_profit']:
                    self.signal(symbol, timestamp, 'sell')
                    del self.positions[symbol]
                    del self.entry_prices[symbol]
                
                # Trailing stop (optional)
                elif price > pos['entry_price']:
                    # Update stop loss to trail the price
                    new_stop = price * 0.95
                    if new_stop > pos['stop_loss']:
                        pos['stop_loss'] = new_stop
        
        def should_enter(self, symbol, price):
            # Your entry logic here
            return True  # Placeholder

Strategy Optimization
---------------------

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import itertools
    from backtester import run_backtest

    def optimize_strategy_parameters():
        """Optimize moving average strategy parameters"""
        
        # Parameter ranges to test
        short_windows = [5, 10, 15, 20]
        long_windows = [20, 30, 40, 50]
        
        best_sharpe = -999
        best_params = None
        results_log = []
        
        for short, long in itertools.product(short_windows, long_windows):
            if short >= long:  # Skip invalid combinations
                continue
            
            # Create strategy class with parameters
            class OptimizedStrategy(MovingAverageCrossStrategy):
                def __init__(self, event_queue, data_handler):
                    super().__init__(event_queue, data_handler, short, long)
            
            # Run backtest
            results = run_backtest(
                data_dict=data_dict,
                strategy_class=OptimizedStrategy,
                initial_cash=100_000.0
            )
            
            sharpe = results.metrics.sharpe_ratio
            results_log.append({
                'short_window': short,
                'long_window': long,
                'sharpe_ratio': sharpe,
                'total_return': results.metrics.total_return,
                'max_drawdown': results.metrics.max_drawdown
            })
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (short, long)
            
            print(f"MA({short},{long}): Sharpe={sharpe:.2f}")
        
        print(f"Best parameters: MA({best_params[0]},{best_params[1]}) with Sharpe={best_sharpe:.2f}")
        
        return results_log, best_params

Walk-Forward Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def walk_forward_analysis(data_dict, strategy_class, window_months=6):
        """Perform walk-forward analysis"""
        
        results = []
        
        # Get date range
        all_dates = []
        for df in data_dict.values():
            all_dates.extend(df['timestamp'].tolist())
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        current_date = start_date
        
        while current_date < end_date:
            # Define training and testing periods
            train_end = current_date + pd.DateOffset(months=window_months)
            test_end = train_end + pd.DateOffset(months=1)
            
            if test_end > end_date:
                break
            
            # Filter data for periods
            test_data = {}
            for symbol, df in data_dict.items():
                test_mask = (df['timestamp'] >= train_end) & (df['timestamp'] < test_end)
                test_data[symbol] = df[test_mask].copy()
            
            # Run backtest on test period
            if all(len(df) > 0 for df in test_data.values()):
                test_results = run_backtest(
                    data_dict=test_data,
                    strategy_class=strategy_class,
                    initial_cash=100_000.0
                )
                
                results.append({
                    'period_start': train_end,
                    'period_end': test_end,
                    'return': test_results.metrics.total_return,
                    'sharpe': test_results.metrics.sharpe_ratio,
                    'max_drawdown': test_results.metrics.max_drawdown
                })
            
            current_date = train_end
        
        return pd.DataFrame(results)

Best Practices
--------------

Strategy Development
~~~~~~~~~~~~~~~~~~~~

1. **Start Simple**: Begin with basic strategies and add complexity gradually
2. **Avoid Overfitting**: Test on out-of-sample data to validate robustness
3. **Consider Transaction Costs**: Include realistic commissions and slippage
4. **Use Proper Risk Management**: Always implement position sizing and risk rules
5. **Validate Assumptions**: Ensure your strategy logic is sound and implementable

Data Quality
~~~~~~~~~~~~

1. **Clean Data**: Remove or handle missing values, outliers, and corporate actions
2. **Survivorship Bias**: Include delisted stocks in historical analysis
3. **Point-in-Time Data**: Use data as it was available at the time
4. **Consistent Frequency**: Ensure all symbols have the same data frequency
5. **Timezone Handling**: Use consistent timezone-aware timestamps

Common Pitfalls
~~~~~~~~~~~~~~~

1. **Look-Ahead Bias**: Using future information in historical analysis
2. **Data Snooping**: Over-optimizing on the same dataset
3. **Ignoring Costs**: Underestimating transaction costs and slippage
4. **Curve Fitting**: Creating overly complex strategies that don't generalize
5. **Insufficient Testing**: Not testing across different market conditions

Next Steps
----------

- Explore :doc:`risk_management` for capital protection
- Check :doc:`examples` for more complex strategy patterns
- Use the :doc:`dashboard` to analyze your strategy performance