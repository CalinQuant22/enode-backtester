"""Reusable UI components."""

from dash import html, dcc
import dash_bootstrap_components as dbc

def MetricCard(title: str, value: str, color: str = "primary"):
    """Create a metric display card."""
    
    return dbc.Card([
        dbc.CardBody([
            html.H4(value, className=f"text-{color} mb-0"),
            html.P(title, className="text-muted mb-0")
        ])
    ], className="text-center")

def create_tabs(portfolio, metrics, monte_carlo):
    """Create main content tabs."""
    
    from .layout import create_performance_tab, create_risk_tab
    
    return dbc.Tabs([
        dbc.Tab(
            label="📈 Performance", 
            tab_id="performance",
            children=[
                html.Div([
                    create_performance_tab(portfolio, metrics)
                ], className="mt-3")
            ]
        ),
        dbc.Tab(
            label="📊 Risk Analysis", 
            tab_id="risk",
            children=[
                html.Div([
                    create_risk_tab(metrics, portfolio)
                ], className="mt-3")
            ]
        ),
        dbc.Tab(
            label="🎯 Trade Analysis", 
            tab_id="trades",
            children=[
                html.Div([
                    create_trade_tab(portfolio, metrics)
                ], className="mt-3")
            ]
        ),
        dbc.Tab(
            label="🎲 Monte Carlo", 
            tab_id="monte_carlo",
            children=[
                html.Div([
                    create_monte_carlo_tab(monte_carlo, portfolio)
                ], className="mt-3")
            ]
        ),
    ], active_tab="performance")

def create_trade_tab(portfolio, metrics):
    """Trade analysis tab content."""
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Trade Statistics"),
                dbc.CardBody([
                    html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Metric", style={"color": "#ffffff"}),
                                html.Th("Value", style={"color": "#ffffff"})
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("Total Trades"), html.Td(str(len(portfolio.trade_log)))]),
                            html.Tr([html.Td("Win Rate"), html.Td(f"{metrics.win_rate:.1%}")]),
                            html.Tr([html.Td("Profit Factor"), html.Td(f"{metrics.profit_factor:.2f}")]),
                            html.Tr([html.Td("Avg Win"), html.Td(f"${metrics.avg_win:.2f}")]),
                            html.Tr([html.Td("Avg Loss"), html.Td(f"${metrics.avg_loss:.2f}")]),
                        ])
                    ], className="table table-striped", style={"color": "#ffffff", "backgroundColor": "#2d2d2d"})
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Recent Trades"),
                dbc.CardBody([
                    create_trades_table(portfolio.trade_log[-10:] if portfolio.trade_log else [])
                ])
            ])
        ], width=6),
    ])

def create_monte_carlo_tab(monte_carlo, portfolio):
    """Monte Carlo analysis tab content."""
    
    if not monte_carlo or "error" in monte_carlo:
        error_msg = monte_carlo.get("error", "Monte Carlo analysis not available") if monte_carlo else "Monte Carlo analysis not available"
        return dbc.Alert(error_msg, color="warning")
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎲 Scenario Analysis"),
                dbc.CardBody([
                    create_scenario_table(monte_carlo["final_value_percentiles"])
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Risk Metrics"),
                dbc.CardBody([
                    html.H4(f"{monte_carlo['probability_of_loss']:.1%}", className="text-danger"),
                    html.P("Probability of Loss", className="text-muted")
                ])
            ])
        ], width=6),
    ])

def create_trades_table(trades):
    """Create trades table."""
    
    if not trades:
        return html.P("No trades to display", className="text-muted")
    
    rows = []
    for trade in trades:
        try:
            rows.append(html.Tr([
                html.Td(str(trade.symbol)),
                html.Td(f"{int(trade.quantity):,}"),
                html.Td(f"${float(trade.fill_price):.2f}"),
                html.Td(str(trade.signal).upper()),
            ], style={"color": "#ffffff"}))
        except (AttributeError, ValueError) as e:
            # Skip malformed trade data
            continue
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Symbol", style={"color": "#ffffff"}),
                html.Th("Quantity", style={"color": "#ffffff"}),
                html.Th("Price", style={"color": "#ffffff"}),
                html.Th("Signal", style={"color": "#ffffff"}),
            ])
        ]),
        html.Tbody(rows)
    ], className="table table-sm", style={"color": "#ffffff", "backgroundColor": "#2d2d2d"})

def create_scenario_table(percentiles):
    """Create Monte Carlo scenario table."""
    
    if not percentiles:
        return html.P("No scenario data available", className="text-muted")
    
    scenarios = [
        ("Best Case", "95%", percentiles.get("95%", 0)),
        ("Good", "75%", percentiles.get("75%", 0)),
        ("Expected", "50%", percentiles.get("50%", 0)),
        ("Poor", "25%", percentiles.get("25%", 0)),
        ("Worst Case", "5%", percentiles.get("5%", 0)),
    ]
    
    rows = []
    for name, pct, value in scenarios:
        rows.append(html.Tr([
            html.Td(name),
            html.Td(pct),
            html.Td(f"${value:,.0f}"),
        ], style={"color": "#ffffff"}))
    
    return html.Table([
        html.Thead([
            html.Tr([
                html.Th("Scenario", style={"color": "#ffffff"}),
                html.Th("Percentile", style={"color": "#ffffff"}),
                html.Th("Final Value", style={"color": "#ffffff"}),
            ])
        ]),
        html.Tbody(rows)
    ], className="table table-striped", style={"color": "#ffffff", "backgroundColor": "#2d2d2d"})