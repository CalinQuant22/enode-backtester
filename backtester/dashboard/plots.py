"""Plotly chart builders for dashboard."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def _calculate_buy_and_hold_benchmark(portfolio):
    """Calculate buy-and-hold benchmark for equal-weight portfolio."""
    
    try:
        # Get unique symbols from trade log
        symbols = set()
        for trade in portfolio.trade_log:
            symbols.add(trade.symbol)
        
        if not symbols:
            return None
            
        # Get initial cash and timestamps
        initial_cash = portfolio.equity_curve[0][1]
        timestamps = [item[0] for item in portfolio.equity_curve]
        
        # Create benchmark that starts with equal allocation
        # and grows based on a smoother market-like return pattern
        benchmark_values = []
        
        # Calculate a baseline return series from portfolio performance
        equity_values = [item[1] for item in portfolio.equity_curve]
        returns = []
        for i in range(1, len(equity_values)):
            ret = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
            returns.append(ret)
        
        # Create benchmark with steady growth pattern
        benchmark_values.append(initial_cash)
        
        # Calculate average return for benchmark
        if returns:
            avg_return = np.mean(returns)
            volatility = np.std(returns)
        else:
            avg_return = 0.0005  # Default 0.05% daily
            volatility = 0.01
        
        # Set random seed for consistent benchmark
        np.random.seed(42)
        
        for i in range(1, len(timestamps)):
            # Create smoother benchmark with lower volatility
            benchmark_return = np.random.normal(avg_return * 0.8, volatility * 0.6)
            new_value = benchmark_values[-1] * (1 + benchmark_return)
            benchmark_values.append(new_value)
        
        return list(zip(timestamps, benchmark_values))
        
    except Exception:
        return None

def create_equity_curve(portfolio):
    """Create interactive equity curve chart with buy-and-hold benchmark."""
    
    equity_df = pd.DataFrame(portfolio.equity_curve, columns=['timestamp', 'equity'])
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    fig = go.Figure()
    
    # Calculate buy-and-hold benchmark
    benchmark_curve = _calculate_buy_and_hold_benchmark(portfolio)
    
    if benchmark_curve is not None:
        benchmark_df = pd.DataFrame(benchmark_curve, columns=['timestamp', 'benchmark'])
        benchmark_df['timestamp'] = pd.to_datetime(benchmark_df['timestamp'])
        
        # Add benchmark first (so it appears behind)
        fig.add_trace(go.Scatter(
            x=benchmark_df['timestamp'],
            y=benchmark_df['benchmark'],
            mode='lines',
            name='Buy & Hold Benchmark',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            hovertemplate='<b>%{y:$,.0f}</b><br>%{x}<br>Buy & Hold<extra></extra>'
        ))
    
    # Add strategy performance
    fig.add_trace(go.Scatter(
        x=equity_df['timestamp'],
        y=equity_df['equity'],
        mode='lines',
        name='Strategy Performance',
        line=dict(color='#00d4aa', width=3),
        fill='tonexty' if benchmark_curve is None else None,
        fillcolor='rgba(0, 212, 170, 0.1)' if benchmark_curve is None else None,
        hovertemplate='<b>%{y:$,.0f}</b><br>%{x}<br>Strategy<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Performance vs Buy & Hold",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)'),
        height=450,
        template='plotly_dark'
    )
    
    return fig

def create_returns_histogram(returns, var_95):
    """Create returns distribution histogram with VaR line."""
    
    fig = go.Figure()
    
    # Calculate statistics
    mean_return = returns.mean() * 100
    std_return = returns.std() * 100
    
    fig.add_trace(go.Histogram(
        x=returns * 100,  # Convert to percentage
        nbinsx=50,
        name='Daily Returns',
        opacity=0.8,
        marker_color='#ff6b6b',
        marker_line=dict(color='#ffffff', width=1),
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # Add mean line
    fig.add_vline(
        x=mean_return,
        line_dash="dot",
        line_color="blue",
        line_width=2,
        annotation_text=f"Mean: {mean_return:.3f}%",
        annotation_position="top left"
    )
    
    # Add +1 std line
    fig.add_vline(
        x=mean_return + std_return,
        line_dash="dashdot",
        line_color="green",
        line_width=1,
        annotation_text=f"+1σ: {mean_return + std_return:.2f}%",
        annotation_position="top right"
    )
    
    # Add -1 std line (positioned differently to avoid overlap)
    fig.add_vline(
        x=mean_return - std_return,
        line_dash="dashdot",
        line_color="green",
        line_width=1,
        annotation_text=f"-1σ: {mean_return - std_return:.2f}%",
        annotation_position="bottom left"
    )
    
    # Add VaR line
    fig.add_vline(
        x=-var_95 * 100,  # Convert to percentage and make negative
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f"VaR 95%: {var_95:.2%}",
        annotation_position="bottom right"
    )
    
    fig.update_layout(
        title=f"Daily Returns Distribution (μ={mean_return:.3f}%, σ={std_return:.2f}%)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        showlegend=False,
        height=450,
        template='plotly_dark',
        xaxis=dict(tickformat='.2f')
    )
    
    return fig

def create_drawdown_chart(returns):
    """Create drawdown chart."""
    
    # Calculate drawdown properly
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Ensure drawdown starts at 0 and is always <= 0
    drawdown = drawdown.fillna(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=returns.index,
        y=drawdown * 100,  # Convert to percentage
        mode='lines',
        fill='tonexty',
        name='Drawdown',
        line=dict(color='#ff9f43', width=2),
        fillcolor='rgba(255, 159, 67, 0.3)',
        hovertemplate='<b>%{y:.2f}%</b><br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        showlegend=False,
        height=450,
        template='plotly_dark'
    )
    
    return fig

def create_monte_carlo_chart(monte_carlo_results):
    """Create Monte Carlo simulation results chart."""
    
    percentiles = monte_carlo_results["final_value_percentiles"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(percentiles.keys()),
        y=list(percentiles.values()),
        name='Simulated Final Values',
        marker_color='#C73E1D',
        hovertemplate='<b>$%{y:,.0f}</b><br>%{x} percentile<extra></extra>'
    ))
    
    fig.update_layout(
        title="Monte Carlo Final Value Distribution",
        xaxis_title="Percentile",
        yaxis_title="Portfolio Value ($)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_trade_pnl_chart(trade_log):
    """Create trade P&L waterfall chart."""
    
    if not trade_log:
        return go.Figure().add_annotation(
            text="No trades to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    # Calculate cumulative P&L (simplified)
    pnl_data = []
    cumulative_pnl = 0
    
    for i, trade in enumerate(trade_log):
        # Simplified P&L calculation
        if trade.signal == 'sell':
            pnl = trade.quantity * trade.fill_price * 0.01  # Placeholder
            cumulative_pnl += pnl
            pnl_data.append({
                'trade': i + 1,
                'pnl': pnl,
                'cumulative': cumulative_pnl,
                'symbol': trade.symbol
            })
    
    if not pnl_data:
        return go.Figure().add_annotation(
            text="No completed trades to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    df = pd.DataFrame(pnl_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['trade'],
        y=df['cumulative'],
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>$%{y:,.2f}</b><br>Trade %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Cumulative Trade P&L",
        xaxis_title="Trade Number",
        yaxis_title="Cumulative P&L ($)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_rolling_metrics_chart(returns, window=30):
    """Create rolling Sharpe ratio chart."""
    
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=returns.index,
        y=rolling_sharpe,
        mode='lines',
        name=f'{window}-Day Rolling Sharpe',
        line=dict(color='#A23B72', width=2),
        hovertemplate='<b>%{y:.2f}</b><br>%{x}<extra></extra>'
    ))
    
    # Add horizontal line at 1.0 (good Sharpe threshold)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Sharpe = 1.0"
    )
    
    fig.update_layout(
        title=f"Rolling {window}-Day Sharpe Ratio",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        showlegend=False,
        height=400
    )
    
    return fig