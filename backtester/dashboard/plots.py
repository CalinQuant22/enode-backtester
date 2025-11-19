"""Plotly chart builders for dashboard."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_equity_curve(portfolio):
    """Create interactive equity curve chart."""
    
    equity_df = pd.DataFrame(portfolio.equity_curve, columns=['timestamp', 'equity'])
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=equity_df['timestamp'],
        y=equity_df['equity'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00d4aa', width=3),
        fill='tonexty',
        fillcolor='rgba(0, 212, 170, 0.1)',
        hovertemplate='<b>%{y:$,.0f}</b><br>%{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        showlegend=False,
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
    
    # Add +/- 1 std lines
    fig.add_vline(
        x=mean_return + std_return,
        line_dash="dashdot",
        line_color="green",
        line_width=1,
        annotation_text=f"+1σ: {mean_return + std_return:.2f}%",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=mean_return - std_return,
        line_dash="dashdot",
        line_color="green",
        line_width=1,
        annotation_text=f"-1σ: {mean_return - std_return:.2f}%",
        annotation_position="top right"
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
    
    # Calculate drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
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