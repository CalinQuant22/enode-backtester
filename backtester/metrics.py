"""
Advanced strategy performance metrics and statistical analysis.

This module provides comprehensive risk-adjusted performance metrics,
statistical analysis, and confidence intervals for backtesting results.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy import stats


# ---------------------------------------------------------------------
# Performance Metrics Dataclass
# ---------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Comprehensive performance and risk metrics."""

    # Returns
    total_return: float
    annualized_return: float
    volatility: float

    # Risk Metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade Statistics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float

    # Statistical Tests
    return_confidence_interval: Tuple[float, float]
    sharpe_confidence_interval: Tuple[float, float]

    # Additional Metrics
    var_95: float  # Value at Risk (positive number = loss)
    cvar_95: float  # Conditional VaR (expected tail loss)
    skewness: float
    kurtosis: float


# ---------------------------------------------------------------------
# Strategy Analyzer
# ---------------------------------------------------------------------

class StrategyAnalyzer:
    """Advanced strategy performance analyzer with statistical rigor."""

    def __init__(self, portfolio, benchmark_return: float = 0.02):
        """
        Args:
            portfolio: Portfolio object from backtest
            benchmark_return: Annual risk-free rate for Sharpe calculation
        """
        self.portfolio = portfolio
        self.benchmark_return = benchmark_return
        self.returns = self._calculate_returns()
        self.trade_log = self._get_trade_log()

    # -------------------------------------------------------------
    # Return & Trade Log Extraction
    # -------------------------------------------------------------

    def _calculate_returns(self) -> pd.Series:
        """Calculate daily returns from equity curve."""
        equity_df = pd.DataFrame(self.portfolio.equity_curve, columns=['timestamp', 'equity'])
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df = equity_df.set_index('timestamp')
        return equity_df['equity'].pct_change().dropna()

    def _get_trade_log(self) -> pd.DataFrame:
        """Convert portfolio trade log into a DataFrame."""
        if not self.portfolio.trade_log:
            return pd.DataFrame()

        trades = []
        for fill in self.portfolio.trade_log:
            trades.append({
                'timestamp': fill.timestamp,
                'symbol': fill.symbol,
                'quantity': fill.quantity,
                'price': fill.fill_price,
                'signal': fill.signal,
            })
        return pd.DataFrame(trades)


    # -------------------------------------------------------------
    # Core Metrics
    # -------------------------------------------------------------

    def calculate_metrics(self, confidence_level: float = 0.95) -> PerformanceMetrics:
        """"Compute full suite of performance metrics."""

        if len(self.returns) == 0:
            return self._empty_metrics()

        # Basic returns
        equity_df = pd.DataFrame(self.portfolio.equity_curve, columns=['timestamp', 'equity'])
        starting_equity = equity_df['equity'].iloc[0]
        ending_equity = equity_df['equity'].iloc[-1]

        total_return = ending_equity / starting_equity - 1
        annualized_return = (1 + total_return) ** (252 / len(self.returns)) - 1
        volatility = self.returns.std() * np.sqrt(252)

        # Risk-free adjustment for Sharpe/Sortino
        excess_returns = self.returns - self.benchmark_return / 252

        # Sharpe (with numerical stability)
        excess_std = excess_returns.std()
        if excess_std > 1e-8:  # Avoid division by near-zero
            sharpe_ratio = excess_returns.mean() / excess_std * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino (downside risk of excess returns)
        downside_excess = excess_returns[excess_returns < 0]
        sortino_ratio = (
            excess_returns.mean() / downside_excess.std() * np.sqrt(252)
            if len(downside_excess) > 0 and downside_excess.std() > 0
            else 0
        )

        # Drawdown
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade stats
        win_rate, profit_factor, avg_win, avg_loss = self._calculate_trade_stats()

        # Bootstrap confidence intervals
        return_ci = self._bootstrap_confidence_interval(self.returns, np.mean, confidence_level)
        sharpe_ci = self._bootstrap_confidence_interval(excess_returns, self._safe_sharpe_calc, confidence_level)

        # VaR & CVaR (positive numbers = losses)
        var_95 = -np.percentile(self.returns, 5)
        var_threshold = np.percentile(self.returns, 5)  # Original 5th percentile
        tail_losses = self.returns[self.returns <= var_threshold]
        cvar_95 = -tail_losses.mean() if len(tail_losses) > 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            return_confidence_interval=return_ci,
            sharpe_confidence_interval=sharpe_ci,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=stats.skew(self.returns),
            kurtosis=stats.kurtosis(self.returns)  # Fisher (excess) kurtosis
        )


    # -------------------------------------------------------------
    # Trade Statistics
    # -------------------------------------------------------------

    def _calculate_trade_stats(self) -> Tuple[float, float, float, float]:
        """Compute win rate, avg win/loss, profit factor."""
        if self.trade_log.empty:
            return 0.0, 0.0, 0.0, 0.0

        pnl_by_trade = []
        positions = {}

        for _, trade in self.trade_log.iterrows():
            sym = trade['symbol']
            qty = trade['quantity']
            price = trade['price'] if 'price' in trade else trade['fill_price']
            signal = trade['signal']

            if sym not in positions:
                positions[sym] = {'qty': 0, 'avg_price': 0}

            # Buy (increase position)
            if signal == 'buy':
                old_qty = positions[sym]['qty']
                new_qty = old_qty + qty
                if old_qty == 0:
                    positions[sym]['avg_price'] = price
                else:
                    positions[sym]['avg_price'] = (
                        old_qty * positions[sym]['avg_price'] + qty * price
                    ) / new_qty
                positions[sym]['qty'] = new_qty

            # Sell (reduce position)
            elif signal == 'sell' and positions[sym]['qty'] > 0:
                sell_qty = min(qty, positions[sym]['qty'])
                pnl = sell_qty * (price - positions[sym]['avg_price'])
                pnl_by_trade.append(pnl)
                positions[sym]['qty'] -= sell_qty

        if not pnl_by_trade:
            return 0.0, 0.0, 0.0, 0.0

        pnl_series = pd.Series(pnl_by_trade)
        wins = pnl_series[pnl_series > 0]
        losses = pnl_series[pnl_series < 0]

        win_rate = len(wins) / len(pnl_series)
        profit_factor = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float('inf')

        return win_rate, profit_factor, wins.mean() if len(wins) else 0, losses.mean() if len(losses) else 0


    # -------------------------------------------------------------
    # Statistical Utilities
    # -------------------------------------------------------------

    def _safe_sharpe_calc(self, sample: np.ndarray) -> float:
        """Stable Sharpe calculation for bootstrap samples."""
        if len(sample) < 2:
            return 0.0
        std_val = np.std(sample)
        if std_val < 1e-8:
            return 0.0
        return np.mean(sample) / std_val * np.sqrt(252)

    def _bootstrap_confidence_interval(self, data: pd.Series, stat_func, confidence: float, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Generic bootstrap CI calculator."""
        if len(data) < 2:
            return (0.0, 0.0)

        stats_ = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            val = stat_func(sample)
            if not np.isnan(val) and not np.isinf(val):
                stats_.append(val)

        if not stats_:
            return (0.0, 0.0)

        alpha = 1 - confidence
        return (
            np.percentile(stats_, 100 * alpha / 2),
            np.percentile(stats_, 100 * (1 - alpha / 2))
        )


    # -------------------------------------------------------------
    # Empty Output
    # -------------------------------------------------------------

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return zero metrics for empty result."""
        return PerformanceMetrics(
            total_return=0.0, annualized_return=0.0, volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
            win_rate=0.0, profit_factor=0.0, avg_win=0.0, avg_loss=0.0,
            return_confidence_interval=(0.0, 0.0), sharpe_confidence_interval=(0.0, 0.0),
            var_95=0.0, cvar_95=0.0, skewness=0.0, kurtosis=0.0
        )


    # -------------------------------------------------------------
    # Monte Carlo Simulation
    # -------------------------------------------------------------

    def monte_carlo_analysis(self, n_simulations: int = 1000) -> Dict:
        """Simulate possible future equity outcomes using normal-distribution returns."""

        if len(self.returns) < 10:
            return {"error": "Insufficient data for Monte Carlo analysis"}

        # Fit normal distribution to returns
        mu, sigma = stats.norm.fit(self.returns)

        final_values = []
        max_drawdowns = []

        initial_value = self.portfolio.equity_curve[0][1]
        n_periods = len(self.returns)

        for _ in range(n_simulations):
            sim_returns = np.random.normal(mu, sigma, n_periods)
            equity_curve = initial_value * (1 + sim_returns).cumprod()
            final_values.append(equity_curve[-1])

            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_drawdowns.append(drawdown.min())

        return {
            "final_value_percentiles": {
                "5%": np.percentile(final_values, 5),
                "25%": np.percentile(final_values, 25),
                "50%": np.percentile(final_values, 50),
                "75%": np.percentile(final_values, 75),
                "95%": np.percentile(final_values, 95)
            },
            "max_drawdown_percentiles": {
                "5%": np.percentile(max_drawdowns, 5),
                "25%": np.percentile(max_drawdowns, 25),
                "50%": np.percentile(max_drawdowns, 50),
                "75%": np.percentile(max_drawdowns, 75),
                "95%": np.percentile(max_drawdowns, 95)
            },
            "probability_of_loss": sum(1 for v in final_values if v < initial_value) / len(final_values)
        }


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def analyze_strategy(portfolio, benchmark_return: float = 0.02) -> Tuple[PerformanceMetrics, Dict]:
    """Convenience function for strategy performance analysis."""
    analyzer = StrategyAnalyzer(portfolio, benchmark_return)
    return analyzer.calculate_metrics(), analyzer.monte_carlo_analysis()
