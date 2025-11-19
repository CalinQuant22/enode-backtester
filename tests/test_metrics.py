"""
Comprehensive tests for the metrics module.

Tests are organized by functionality and test both normal cases and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock

from backtester.metrics import StrategyAnalyzer, PerformanceMetrics, analyze_strategy


class TestStrategyAnalyzer:
    """Test the StrategyAnalyzer class functionality."""
    
    def test_empty_portfolio_returns_zero_metrics(self):
        """Test that empty portfolio returns all zero metrics."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000)]
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        metrics = analyzer.calculate_metrics()
        
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.win_rate == 0.0
    
    def test_calculate_returns_from_equity_curve(self):
        """Test that returns are calculated correctly from equity curve."""
        # Create portfolio with simple equity progression: 100k -> 110k -> 121k
        dates = [datetime(2023, 1, 1), datetime(2023, 1, 2), datetime(2023, 1, 3)]
        equity_values = [100000, 110000, 121000]
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        
        # Expected returns: [0.10, 0.10] (10% each day)
        expected_values = [0.10, 0.10]
        assert len(analyzer.returns) == 2
        assert analyzer.returns.iloc[0] == pytest.approx(0.10)
        assert analyzer.returns.iloc[1] == pytest.approx(0.10)
    
    def test_total_return_calculation(self):
        """Test total return calculation is correct."""
        dates = [datetime(2023, 1, 1), datetime(2023, 1, 2)]
        portfolio = Mock()
        portfolio.equity_curve = [(dates[0], 100000), (dates[1], 120000)]
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        metrics = analyzer.calculate_metrics()
        
        # 120k / 100k - 1 = 0.20 (20% return)
        assert metrics.total_return == pytest.approx(0.20)
    
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio uses excess returns correctly."""
        # Create returns of 1% daily with 2% annual risk-free rate
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        daily_returns = [0.01] * 9  # 1% daily return
        equity_values = [100000 * (1.01 ** i) for i in range(10)]
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio, benchmark_return=0.02)
        metrics = analyzer.calculate_metrics()
        
        # With constant returns, volatility approaches zero
        # The Sharpe calculation should handle this edge case
        # Either return 0 or a very large number due to near-zero std
        assert metrics.volatility < 0.01  # Very low volatility
        # Sharpe could be 0 (our safe calc) or very large (numerical)
        assert metrics.sharpe_ratio == 0.0 or abs(metrics.sharpe_ratio) > 1000
    
    def test_sharpe_ratio_with_volatility(self):
        """Test Sharpe ratio with actual volatility."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        # Create varying returns: alternating 2% and -1%
        returns = [0.02 if i % 2 == 0 else -0.01 for i in range(99)]
        
        equity_values = [100000]
        for ret in returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio, benchmark_return=0.0)  # Zero risk-free rate
        metrics = analyzer.calculate_metrics()
        
        # With zero risk-free rate, excess returns = raw returns
        # Mean return = (0.02 + (-0.01)) / 2 = 0.005
        # Should have positive Sharpe ratio
        assert metrics.sharpe_ratio > 0
    
    def test_sortino_ratio_uses_downside_deviation(self):
        """Test Sortino ratio uses only negative excess returns for denominator."""
        dates = pd.date_range('2023-01-01', periods=6, freq='D')
        # Returns: [0.05, -0.02, 0.03, -0.01, 0.04] (mix of positive/negative)
        returns = [0.05, -0.02, 0.03, -0.01, 0.04]
        
        equity_values = [100000]
        for ret in returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio, benchmark_return=0.0)
        metrics = analyzer.calculate_metrics()
        
        # Should have positive Sortino ratio (positive mean, negative returns for denominator)
        assert metrics.sortino_ratio > 0
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        # Equity: 100k -> 120k -> 110k -> 105k -> 115k
        # Peak at 120k, trough at 105k = 12.5% drawdown
        equity_values = [100000, 120000, 110000, 105000, 115000]
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        metrics = analyzer.calculate_metrics()
        
        # Max drawdown = (105k - 120k) / 120k = -0.125 (-12.5%)
        assert metrics.max_drawdown == pytest.approx(-0.125)
    
    def test_var_and_cvar_calculation(self):
        """Test VaR and CVaR are calculated correctly."""
        # Create portfolio with known return distribution
        dates = pd.date_range('2023-01-01', periods=101, freq='D')
        # Returns from -0.05 to +0.05 in steps of 0.001
        returns = np.linspace(-0.05, 0.05, 100)
        
        equity_values = [100000]
        for ret in returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        metrics = analyzer.calculate_metrics()
        
        # 5th percentile of returns should be around -0.045
        # VaR should be positive (loss convention)
        assert metrics.var_95 > 0
        assert metrics.cvar_95 > 0
        assert metrics.cvar_95 >= metrics.var_95  # CVaR should be >= VaR


class TestTradeStatistics:
    """Test trade-level statistics calculations."""
    
    def create_mock_fill(self, symbol, quantity, price, signal, timestamp=None):
        """Helper to create mock fill events."""
        fill = Mock()
        fill.symbol = symbol
        fill.quantity = quantity
        fill.fill_price = price
        fill.signal = signal
        fill.timestamp = timestamp or datetime.now()
        return fill
    
    def test_simple_profitable_trade(self):
        """Test a simple buy-sell profitable trade."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000), (datetime.now(), 100000)]
        portfolio.trade_log = [
            self.create_mock_fill('AAPL', 100, 50.0, 'buy'),
            self.create_mock_fill('AAPL', 100, 55.0, 'sell')
        ]
        
        analyzer = StrategyAnalyzer(portfolio)
        win_rate, profit_factor, avg_win, avg_loss = analyzer._calculate_trade_stats()
        
        # One trade, profitable: 100 * (55 - 50) = $500 profit
        assert win_rate == 1.0  # 100% win rate
        assert profit_factor == float('inf')  # No losses
        assert avg_win == 500.0
        assert avg_loss == 0.0
    
    def test_simple_losing_trade(self):
        """Test a simple buy-sell losing trade."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000), (datetime.now(), 100000)]
        portfolio.trade_log = [
            self.create_mock_fill('AAPL', 100, 50.0, 'buy'),
            self.create_mock_fill('AAPL', 100, 45.0, 'sell')
        ]
        
        analyzer = StrategyAnalyzer(portfolio)
        win_rate, profit_factor, avg_win, avg_loss = analyzer._calculate_trade_stats()
        
        # One trade, losing: 100 * (45 - 50) = -$500 loss
        assert win_rate == 0.0  # 0% win rate
        assert profit_factor == 0.0  # No wins
        assert avg_win == 0.0
        assert avg_loss == -500.0
    
    def test_mixed_trades(self):
        """Test mixed winning and losing trades."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000), (datetime.now(), 100000)]
        portfolio.trade_log = [
            # Trade 1: Buy 100 @ $50, sell 100 @ $55 = +$500
            self.create_mock_fill('AAPL', 100, 50.0, 'buy'),
            self.create_mock_fill('AAPL', 100, 55.0, 'sell'),
            # Trade 2: Buy 200 @ $60, sell 200 @ $58 = -$400
            self.create_mock_fill('GOOGL', 200, 60.0, 'buy'),
            self.create_mock_fill('GOOGL', 200, 58.0, 'sell')
        ]
        
        analyzer = StrategyAnalyzer(portfolio)
        win_rate, profit_factor, avg_win, avg_loss = analyzer._calculate_trade_stats()
        
        assert win_rate == 0.5  # 50% win rate (1 win, 1 loss)
        assert profit_factor == pytest.approx(500.0 / 400.0)  # 1.25
        assert avg_win == 500.0
        assert avg_loss == -400.0
    
    def test_partial_position_closing(self):
        """Test partial position closing."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000), (datetime.now(), 100000)]
        portfolio.trade_log = [
            # Buy 200 shares @ $50
            self.create_mock_fill('AAPL', 200, 50.0, 'buy'),
            # Sell 100 shares @ $55 (partial close)
            self.create_mock_fill('AAPL', 100, 55.0, 'sell'),
            # Sell remaining 100 shares @ $60
            self.create_mock_fill('AAPL', 100, 60.0, 'sell')
        ]
        
        analyzer = StrategyAnalyzer(portfolio)
        win_rate, profit_factor, avg_win, avg_loss = analyzer._calculate_trade_stats()
        
        # Two separate P&L calculations:
        # Trade 1: 100 * (55 - 50) = +$500
        # Trade 2: 100 * (60 - 50) = +$1000
        assert win_rate == 1.0  # Both trades profitable
        assert avg_win == 750.0  # (500 + 1000) / 2
    
    def test_no_trades_returns_zeros(self):
        """Test that no trades returns zero statistics."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000), (datetime.now(), 100000)]
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        win_rate, profit_factor, avg_win, avg_loss = analyzer._calculate_trade_stats()
        
        assert win_rate == 0.0
        assert profit_factor == 0.0
        assert avg_win == 0.0
        assert avg_loss == 0.0


class TestBootstrapConfidenceIntervals:
    """Test bootstrap confidence interval calculations."""
    
    def test_bootstrap_with_constant_data(self):
        """Test bootstrap CI with constant data (zero variance)."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000), (datetime.now(), 100000)]
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        
        # Create constant data
        constant_data = pd.Series([0.01] * 100)
        
        ci = analyzer._bootstrap_confidence_interval(constant_data, np.mean, 0.95, n_bootstrap=100)
        
        # With constant data, CI should be tight around the mean
        assert ci[0] == pytest.approx(0.01, abs=1e-6)
        assert ci[1] == pytest.approx(0.01, abs=1e-6)
    
    def test_bootstrap_with_insufficient_data(self):
        """Test bootstrap CI with insufficient data."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000), (datetime.now(), 100000)]
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        
        # Single data point
        single_data = pd.Series([0.01])
        ci = analyzer._bootstrap_confidence_interval(single_data, np.mean, 0.95)
        
        assert ci == (0.0, 0.0)
    
    def test_safe_sharpe_calculation(self):
        """Test safe Sharpe calculation handles edge cases."""
        portfolio = Mock()
        portfolio.equity_curve = [(datetime.now(), 100000), (datetime.now(), 100000)]
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        
        # Test with zero volatility
        constant_returns = np.array([0.01] * 100)
        sharpe = analyzer._safe_sharpe_calc(constant_returns)
        assert sharpe == 0.0  # Should return 0 for zero volatility
        
        # Test with insufficient data
        short_data = np.array([0.01])
        sharpe = analyzer._safe_sharpe_calc(short_data)
        assert sharpe == 0.0
        
        # Test with normal data
        normal_returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        sharpe = analyzer._safe_sharpe_calc(normal_returns)
        assert sharpe != 0.0  # Should calculate actual Sharpe


class TestMonteCarloAnalysis:
    """Test Monte Carlo simulation functionality."""
    
    def test_monte_carlo_with_insufficient_data(self):
        """Test Monte Carlo returns error with insufficient data."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')  # Only 4 returns
        equity_values = [100000, 101000, 102000, 103000, 104000]
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        result = analyzer.monte_carlo_analysis()
        
        assert "error" in result
    
    def test_monte_carlo_with_sufficient_data(self):
        """Test Monte Carlo produces reasonable results."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        # Create some realistic returns
        np.random.seed(42)  # For reproducible tests
        returns = np.random.normal(0.001, 0.02, 49)  # 0.1% mean, 2% std daily
        
        equity_values = [100000]
        for ret in returns:
            equity_values.append(equity_values[-1] * (1 + ret))
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        analyzer = StrategyAnalyzer(portfolio)
        result = analyzer.monte_carlo_analysis(n_simulations=100)
        
        # Should have proper structure
        assert "final_value_percentiles" in result
        assert "max_drawdown_percentiles" in result
        assert "probability_of_loss" in result
        
        # Percentiles should be ordered
        percentiles = result["final_value_percentiles"]
        assert percentiles["5%"] <= percentiles["25%"] <= percentiles["50%"] <= percentiles["75%"] <= percentiles["95%"]
        
        # Probability should be between 0 and 1
        assert 0 <= result["probability_of_loss"] <= 1


class TestAnalyzeStrategyFunction:
    """Test the convenience function."""
    
    def test_analyze_strategy_returns_tuple(self):
        """Test that analyze_strategy returns metrics and monte carlo results."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        equity_values = [100000 * (1.001 ** i) for i in range(30)]  # Steady growth
        
        portfolio = Mock()
        portfolio.equity_curve = list(zip(dates, equity_values))
        portfolio.trade_log = []
        
        metrics, monte_carlo = analyze_strategy(portfolio)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert isinstance(monte_carlo, dict)
        assert metrics.total_return > 0  # Should be profitable