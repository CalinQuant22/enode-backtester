Examples and Use Cases
======================

This section provides comprehensive examples demonstrating various aspects of the backtesting framework, from simple strategies to advanced multi-asset portfolios.

Complete Workflow Example
-------------------------

Here's a complete example showing the entire workflow from data preparation to dashboard analysis:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from backtester import run_backtest, FixedSizeSizer, get_trade_log_df
    from backtester.strategy import BaseStrategy
    from backtester.event import StockEvent
    from backtester.risk import RiskManager, MaxPositionSizeRule, MaxCashUsageRule

    # 1. Create sample data for multiple assets
    def create_sample_data(symbol, start_date, end_date, initial_price=100):
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Generate realistic price movements
        np.random.seed(hash(symbol) % 2**32)  # Different seed per symbol
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = initial_price * (1 + returns).cumprod()
        
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            prev_close = prices[i-1] if i > 0 else close
            gap = np.random.normal(0, 0.005)
            open_price = prev_close * (1 + gap)
            
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.normal(1000000, 200000))
            
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': max(volume, 100000)
            })
        
        return pd.DataFrame(data)

    # 2. Define a momentum strategy
    class MomentumStrategy(BaseStrategy):
        def __init__(self, event_queue, data_handler, lookback=20, threshold=0.05):
            super().__init__(event_queue, data_handler)
            self.lookback = lookback
            self.threshold = threshold
            self.price_history = {}
            self.positions = {}
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            timestamp = event.payload.timestamp
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
            
            # Need sufficient history
            if len(self.price_history[symbol]) < self.lookback + 1:
                return
            
            # Calculate momentum
            current_price = self.price_history[symbol][-1]
            past_price = self.price_history[symbol][-self.lookback-1]
            momentum = (current_price - past_price) / past_price
            
            current_position = self.positions.get(symbol, 'FLAT')
            
            # Entry signals
            if momentum > self.threshold and current_position == 'FLAT':
                self.signal(symbol, timestamp, 'buy')
                self.positions[symbol] = 'LONG'
            
            # Exit signals
            elif momentum < -self.threshold and current_position == 'LONG':
                self.signal(symbol, timestamp, 'sell')
                self.positions[symbol] = 'FLAT'

    # 3. Prepare multi-asset dataset
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    data_dict = {}
    
    for symbol in symbols:
        data_dict[symbol] = create_sample_data(
            symbol, '2023-01-01', '2023-12-31', 
            initial_price=np.random.uniform(50, 300)
        )

    # 4. Set up risk management
    risk_manager = RiskManager([
        MaxPositionSizeRule(max_position_pct=0.25),  # Max 25% per position
        MaxCashUsageRule(reserve_cash=10000.0)       # Keep $10k reserve
    ])

    # 5. Run backtest
    results = run_backtest(
        data_dict=data_dict,
        strategy_class=MomentumStrategy,
        initial_cash=200_000.0,
        sizer=FixedSizeSizer(50),
        risk_manager=risk_manager
    )

    # 6. Analyze results
    print("=== BACKTEST RESULTS ===")
    print(f"Total Return: {results.metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.metrics.max_drawdown:.2%}")
    print(f"Win Rate: {results.metrics.win_rate:.1%}")
    print(f"Profit Factor: {results.metrics.profit_factor:.2f}")

    # 7. Trade analysis
    trades_df = get_trade_log_df(results.portfolio)
    print(f"\\nTotal Trades: {len(trades_df)}")
    print(f"Symbols Traded: {trades_df['symbol'].nunique()}")
    
    # Performance by symbol
    symbol_performance = trades_df.groupby('symbol').agg({
        'quantity': 'sum',
        'fill_price': 'mean',
        'commission': 'sum'
    }).round(2)
    print("\\nPerformance by Symbol:")
    print(symbol_performance)

    # 8. Launch dashboard for detailed analysis
    results.dashboard()

Mean Reversion Strategy
-----------------------

.. code-block:: python

    class BollingerBandsMeanReversion(BaseStrategy):
        """Mean reversion strategy using Bollinger Bands"""
        
        def __init__(self, event_queue, data_handler, window=20, num_std=2.0):
            super().__init__(event_queue, data_handler)
            self.window = window
            self.num_std = num_std
            self.price_history = {}
            self.positions = {}
        
        def calculate_bollinger_bands(self, prices):
            if len(prices) < self.window:
                return None, None, None
            
            recent_prices = prices[-self.window:]
            sma = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            upper_band = sma + (self.num_std * std)
            lower_band = sma - (self.num_std * std)
            
            return upper_band, sma, lower_band
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            timestamp = event.payload.timestamp
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
            
            # Calculate Bollinger Bands
            upper, middle, lower = self.calculate_bollinger_bands(
                self.price_history[symbol]
            )
            
            if upper is None:
                return
            
            current_position = self.positions.get(symbol, 'FLAT')
            
            # Mean reversion signals
            if price < lower and current_position == 'FLAT':
                # Price below lower band - buy (expect reversion up)
                self.signal(symbol, timestamp, 'buy')
                self.positions[symbol] = 'LONG'
            
            elif price > upper and current_position == 'LONG':
                # Price above upper band - sell (expect reversion down)
                self.signal(symbol, timestamp, 'sell')
                self.positions[symbol] = 'FLAT'
            
            elif price > middle and current_position == 'LONG':
                # Price back to middle - take profit
                self.signal(symbol, timestamp, 'sell')
                self.positions[symbol] = 'FLAT'

Multi-Strategy Portfolio
------------------------

.. code-block:: python

    class MultiStrategyPortfolio:
        """Combine multiple strategies with different allocations"""
        
        def __init__(self, strategies_config):
            self.strategies_config = strategies_config
            self.results = {}
        
        def run_all_strategies(self, data_dict, initial_cash):
            total_weight = sum(config['weight'] for config in self.strategies_config.values())
            
            for name, config in self.strategies_config.items():
                # Allocate cash proportionally
                strategy_cash = initial_cash * (config['weight'] / total_weight)
                
                # Run individual strategy
                results = run_backtest(
                    data_dict=data_dict,
                    strategy_class=config['strategy_class'],
                    initial_cash=strategy_cash,
                    sizer=config.get('sizer', FixedSizeSizer(50)),
                    risk_manager=config.get('risk_manager')
                )
                
                self.results[name] = results
                print(f"{name}: Return={results.metrics.total_return:.2%}, "
                      f"Sharpe={results.metrics.sharpe_ratio:.2f}")
            
            return self.combine_results()
        
        def combine_results(self):
            """Combine individual strategy results"""
            combined_equity = {}
            all_trades = []
            
            # Combine equity curves
            for name, results in self.results.items():
                weight = self.strategies_config[name]['weight']
                
                for timestamp, equity in results.portfolio.equity_curve:
                    if timestamp not in combined_equity:
                        combined_equity[timestamp] = 0
                    combined_equity[timestamp] += equity * weight
                
                # Add strategy name to trades
                for trade in results.portfolio.trade_log:
                    trade_copy = trade.copy()
                    trade_copy['strategy'] = name
                    all_trades.append(trade_copy)
            
            # Convert to sorted list
            combined_equity_list = sorted(combined_equity.items())
            
            return {
                'combined_equity': combined_equity_list,
                'all_trades': all_trades,
                'individual_results': self.results
            }

    # Example usage
    strategies_config = {
        'momentum': {
            'strategy_class': MomentumStrategy,
            'weight': 0.4,
            'sizer': FixedSizeSizer(75)
        },
        'mean_reversion': {
            'strategy_class': BollingerBandsMeanReversion,
            'weight': 0.3,
            'sizer': FixedSizeSizer(50)
        },
        'buy_hold': {
            'strategy_class': BuyAndHoldStrategy,
            'weight': 0.3,
            'sizer': FixedSizeSizer(100)
        }
    }

    multi_portfolio = MultiStrategyPortfolio(strategies_config)
    combined_results = multi_portfolio.run_all_strategies(data_dict, 300_000.0)

Sector Rotation Strategy
------------------------

.. code-block:: python

    class SectorRotationStrategy(BaseStrategy):
        """Rotate between sectors based on relative strength"""
        
        def __init__(self, event_queue, data_handler, sector_map, lookback=60):
            super().__init__(event_queue, data_handler)
            self.sector_map = sector_map  # {symbol: sector}
            self.lookback = lookback
            self.price_history = {}
            self.current_sector = None
            self.sector_performance = {}
        
        def calculate_sector_momentum(self):
            """Calculate momentum for each sector"""
            sector_returns = {}
            
            for symbol, sector in self.sector_map.items():
                if (symbol in self.price_history and 
                    len(self.price_history[symbol]) >= self.lookback + 1):
                    
                    prices = self.price_history[symbol]
                    current_price = prices[-1]
                    past_price = prices[-self.lookback-1]
                    momentum = (current_price - past_price) / past_price
                    
                    if sector not in sector_returns:
                        sector_returns[sector] = []
                    sector_returns[sector].append(momentum)
            
            # Average momentum by sector
            sector_avg_momentum = {}
            for sector, returns in sector_returns.items():
                sector_avg_momentum[sector] = np.mean(returns)
            
            return sector_avg_momentum
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            timestamp = event.payload.timestamp
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
            
            # Only rebalance monthly (simplified)
            if timestamp.day != 1:  # First day of month
                return
            
            # Calculate sector momentum
            sector_momentum = self.calculate_sector_momentum()
            
            if not sector_momentum:
                return
            
            # Find best performing sector
            best_sector = max(sector_momentum.keys(), 
                            key=lambda s: sector_momentum[s])
            
            # If sector changed, rebalance
            if best_sector != self.current_sector:
                # Sell all current positions
                if self.current_sector:
                    for sym, sect in self.sector_map.items():
                        if sect == self.current_sector:
                            self.signal(sym, timestamp, 'sell')
                
                # Buy new sector (equal weight)
                new_sector_symbols = [sym for sym, sect in self.sector_map.items() 
                                    if sect == best_sector]
                
                for sym in new_sector_symbols:
                    self.signal(sym, timestamp, 'buy')
                
                self.current_sector = best_sector
                print(f"Rotated to sector: {best_sector} on {timestamp.date()}")

    # Example with sector mapping
    sector_map = {
        'AAPL': 'Technology',
        'GOOGL': 'Technology', 
        'MSFT': 'Technology',
        'JPM': 'Financial',
        'BAC': 'Financial',
        'XOM': 'Energy',
        'CVX': 'Energy'
    }

Pairs Trading Strategy
----------------------

.. code-block:: python

    class StatisticalArbitragePairs(BaseStrategy):
        """Statistical arbitrage between correlated pairs"""
        
        def __init__(self, event_queue, data_handler, pairs, lookback=30, entry_z=2.0, exit_z=0.5):
            super().__init__(event_queue, data_handler)
            self.pairs = pairs  # [(symbol1, symbol2), ...]
            self.lookback = lookback
            self.entry_z = entry_z
            self.exit_z = exit_z
            
            self.price_history = {}
            self.spread_history = {}
            self.positions = {}  # {pair: 'LONG_SPREAD', 'SHORT_SPREAD', 'FLAT'}
        
        def calculate_z_score(self, spread_history):
            if len(spread_history) < self.lookback:
                return None
            
            recent_spreads = spread_history[-self.lookback:]
            mean_spread = np.mean(recent_spreads)
            std_spread = np.std(recent_spreads)
            
            if std_spread == 0:
                return None
            
            current_spread = spread_history[-1]
            z_score = (current_spread - mean_spread) / std_spread
            
            return z_score
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            timestamp = event.payload.timestamp
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
            
            # Check all pairs involving this symbol
            for pair in self.pairs:
                symbol1, symbol2 = pair
                
                if symbol not in pair:
                    continue
                
                # Need data for both symbols
                if (symbol1 not in self.price_history or 
                    symbol2 not in self.price_history or
                    len(self.price_history[symbol1]) == 0 or
                    len(self.price_history[symbol2]) == 0):
                    continue
                
                # Calculate spread (price ratio)
                price1 = self.price_history[symbol1][-1]
                price2 = self.price_history[symbol2][-1]
                spread = price1 / price2
                
                # Update spread history
                if pair not in self.spread_history:
                    self.spread_history[pair] = []
                self.spread_history[pair].append(spread)
                
                # Calculate z-score
                z_score = self.calculate_z_score(self.spread_history[pair])
                
                if z_score is None:
                    continue
                
                current_position = self.positions.get(pair, 'FLAT')
                
                # Entry signals
                if z_score > self.entry_z and current_position == 'FLAT':
                    # Spread too high - short spread (sell symbol1, buy symbol2)
                    self.signal(symbol1, timestamp, 'sell')
                    self.signal(symbol2, timestamp, 'buy')
                    self.positions[pair] = 'SHORT_SPREAD'
                
                elif z_score < -self.entry_z and current_position == 'FLAT':
                    # Spread too low - long spread (buy symbol1, sell symbol2)
                    self.signal(symbol1, timestamp, 'buy')
                    self.signal(symbol2, timestamp, 'sell')
                    self.positions[pair] = 'LONG_SPREAD'
                
                # Exit signals
                elif (abs(z_score) < self.exit_z and current_position != 'FLAT'):
                    # Spread normalized - close positions
                    if current_position == 'SHORT_SPREAD':
                        self.signal(symbol1, timestamp, 'buy')   # Cover short
                        self.signal(symbol2, timestamp, 'sell')  # Close long
                    elif current_position == 'LONG_SPREAD':
                        self.signal(symbol1, timestamp, 'sell')  # Close long
                        self.signal(symbol2, timestamp, 'buy')   # Cover short
                    
                    self.positions[pair] = 'FLAT'

    # Example pairs
    pairs = [
        ('AAPL', 'MSFT'),   # Tech pair
        ('JPM', 'BAC'),     # Bank pair
        ('XOM', 'CVX')      # Energy pair
    ]

Advanced Risk Management Example
--------------------------------

.. code-block:: python

    from backtester.risk import BaseRiskRule, RiskCheckResult

    class AdvancedRiskStrategy(BaseStrategy):
        """Strategy with integrated risk management"""
        
        def __init__(self, event_queue, data_handler):
            super().__init__(event_queue, data_handler)
            self.price_history = {}
            self.positions = {}
            self.entry_prices = {}
            self.stop_losses = {}
            self.take_profits = {}
        
        def on_stock_event(self, event: StockEvent):
            symbol = event.payload.symbol
            price = event.payload.close
            timestamp = event.payload.timestamp
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
            
            # Check existing positions for risk management
            if symbol in self.positions and self.positions[symbol] == 'LONG':
                entry_price = self.entry_prices[symbol]
                stop_loss = self.stop_losses[symbol]
                take_profit = self.take_profits[symbol]
                
                # Stop loss
                if price <= stop_loss:
                    self.signal(symbol, timestamp, 'sell')
                    self._close_position(symbol)
                    print(f"Stop loss triggered for {symbol} at {price:.2f}")
                
                # Take profit
                elif price >= take_profit:
                    self.signal(symbol, timestamp, 'sell')
                    self._close_position(symbol)
                    print(f"Take profit triggered for {symbol} at {price:.2f}")
                
                # Trailing stop (update stop loss if price moves favorably)
                elif price > entry_price * 1.05:  # 5% profit
                    new_stop = price * 0.95  # 5% trailing stop
                    if new_stop > stop_loss:
                        self.stop_losses[symbol] = new_stop
            
            # Entry logic (simplified momentum)
            elif (len(self.price_history[symbol]) >= 20 and 
                  symbol not in self.positions):
                
                # Calculate momentum
                current_price = self.price_history[symbol][-1]
                past_price = self.price_history[symbol][-20]
                momentum = (current_price - past_price) / past_price
                
                if momentum > 0.05:  # 5% momentum
                    self.signal(symbol, timestamp, 'buy')
                    self._open_position(symbol, price)
        
        def _open_position(self, symbol, entry_price):
            self.positions[symbol] = 'LONG'
            self.entry_prices[symbol] = entry_price
            self.stop_losses[symbol] = entry_price * 0.95  # 5% stop loss
            self.take_profits[symbol] = entry_price * 1.15  # 15% take profit
        
        def _close_position(self, symbol):
            if symbol in self.positions:
                del self.positions[symbol]
                del self.entry_prices[symbol]
                del self.stop_losses[symbol]
                del self.take_profits[symbol]

Performance Comparison Example
------------------------------

.. code-block:: python

    def compare_strategies(data_dict, strategies, initial_cash=100_000):
        """Compare multiple strategies side by side"""
        
        results = {}
        
        for name, strategy_class in strategies.items():
            print(f"Running {name}...")
            
            result = run_backtest(
                data_dict=data_dict,
                strategy_class=strategy_class,
                initial_cash=initial_cash,
                sizer=FixedSizeSizer(50)
            )
            
            results[name] = result
        
        # Create comparison table
        comparison_data = []
        
        for name, result in results.items():
            metrics = result.metrics
            comparison_data.append({
                'Strategy': name,
                'Total Return': f"{metrics.total_return:.2%}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}",
                'Win Rate': f"{metrics.win_rate:.1%}",
                'Profit Factor': f"{metrics.profit_factor:.2f}",
                'Total Trades': len(result.portfolio.trade_log)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\\n=== STRATEGY COMPARISON ===")
        print(comparison_df.to_string(index=False))
        
        return results

    # Example comparison
    strategies_to_compare = {
        'Buy & Hold': BuyAndHoldStrategy,
        'Momentum': MomentumStrategy,
        'Mean Reversion': BollingerBandsMeanReversion,
        'Advanced Risk': AdvancedRiskStrategy
    }

    comparison_results = compare_strategies(data_dict, strategies_to_compare)

Dashboard Integration Examples
------------------------------

.. code-block:: python

    # Example 1: Automated analysis workflow
    def automated_strategy_analysis(data_dict, strategy_class):
        """Run backtest and automatically generate analysis"""
        
        # Run backtest
        results = run_backtest(
            data_dict=data_dict,
            strategy_class=strategy_class,
            initial_cash=100_000.0
        )
        
        # Print summary
        print("=== AUTOMATED ANALYSIS ===")
        print(f"Strategy: {strategy_class.__name__}")
        print(f"Total Return: {results.metrics.total_return:.2%}")
        print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.metrics.max_drawdown:.2%}")
        
        # Save results
        filename = f"{strategy_class.__name__}_results.json"
        results.save(filename)
        print(f"Results saved to: {filename}")
        
        # Launch dashboard
        print("Launching dashboard...")
        results.dashboard()
        
        return results

    # Example 2: Batch analysis with dashboard
    def batch_analysis_with_dashboard(data_dict, strategies):
        """Analyze multiple strategies and create dashboards"""
        
        all_results = []
        
        for name, strategy_class in strategies.items():
            print(f"\\nAnalyzing {name}...")
            
            results = run_backtest(
                data_dict=data_dict,
                strategy_class=strategy_class,
                initial_cash=100_000.0
            )
            
            # Save with descriptive name
            filename = f"{name.lower().replace(' ', '_')}_backtest.json"
            results.save(filename)
            
            all_results.append((name, results, filename))
            
            print(f"  Return: {results.metrics.total_return:.2%}")
            print(f"  Sharpe: {results.metrics.sharpe_ratio:.2f}")
            print(f"  Saved: {filename}")
        
        # Launch best performing strategy dashboard
        best_strategy = max(all_results, key=lambda x: x[1].metrics.sharpe_ratio)
        print(f"\\nLaunching dashboard for best strategy: {best_strategy[0]}")
        best_strategy[1].dashboard()
        
        return all_results

Next Steps
----------

These examples demonstrate the flexibility and power of the backtesting framework. To continue learning:

1. **Modify Examples**: Adapt these examples to your specific use cases
2. **Combine Techniques**: Mix and match different strategy components
3. **Add Complexity**: Gradually increase sophistication as you learn
4. **Use the Dashboard**: Leverage visual analysis for deeper insights
5. **Implement Risk Management**: Always include appropriate risk controls

For more advanced topics, explore:

- Custom data handlers for live feeds
- Advanced execution models with market impact
- Machine learning integration for signal generation
- Multi-asset portfolio optimization
- Real-time monitoring and alerting systems