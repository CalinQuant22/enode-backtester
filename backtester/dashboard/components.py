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
    
    return html.Div([
        # Trade Analysis explanation
        html.Div([
            html.H5([html.I(className="fas fa-info-circle me-2"), "Trade Analysis Overview"]),
            html.P([
                "This section analyzes your individual trades to understand strategy effectiveness. ",
                "Win Rate shows the percentage of profitable trades, while Profit Factor compares ",
                "total gains to total losses. A Profit Factor > 1.0 means your strategy is profitable overall. ",
                "Average Win/Loss helps you understand the typical size of your gains versus losses."
            ])
        ], className="explanation"),
        
        dbc.Row([
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
                        ], className="table table-striped", style={"color": "#ffffff", "backgroundColor": "#2d2d2d"}),
                        
                        # Trade metrics explanations
                        html.Div([
                            html.H6("📖 Metric Explanations", className="mt-3 mb-2"),
                            html.Ul([
                                html.Li([html.Strong("Win Rate: "), "Percentage of trades that were profitable"]),
                                html.Li([html.Strong("Profit Factor: "), "Total gains ÷ Total losses (>1.0 = profitable)"]),
                                html.Li([html.Strong("Avg Win/Loss: "), "Average profit per winning/losing trade"])
                            ], style={"font-size": "0.85rem", "color": "#cccccc"})
                        ], className="explanation")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📋 Recent Trades"),
                    dbc.CardBody([
                        create_trades_table(portfolio.trade_log[-10:] if portfolio.trade_log else []),
                        html.P(f"Showing last 10 of {len(portfolio.trade_log)} total trades", 
                               className="text-muted mt-2", style={"font-size": "0.9rem"})
                    ])
                ])
            ], width=6),
        ])
    ])

def create_monte_carlo_tab(monte_carlo, portfolio):
    """Monte Carlo analysis tab content."""
    
    if not monte_carlo or "error" in monte_carlo:
        error_msg = monte_carlo.get("error", "Monte Carlo analysis not available") if monte_carlo else "Monte Carlo analysis not available"
        return dbc.Alert(error_msg, color="warning")
    
    return html.Div([
        # Monte Carlo explanation
        html.Div([
            html.H5([html.I(className="fas fa-info-circle me-2"), "Monte Carlo Simulation"]),
            html.P([
                "Monte Carlo analysis runs thousands of simulations to estimate possible future outcomes ",
                "based on your strategy's historical return patterns. This helps you understand the range ",
                "of potential results and assess the probability of different scenarios. The percentiles ",
                "show what you might expect in best-case, typical, and worst-case situations."
            ])
        ], className="explanation"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎲 Scenario Analysis"),
                    dbc.CardBody([
                        create_scenario_table(monte_carlo["final_value_percentiles"]),
                        
                        # Scenario explanations
                        html.Div([
                            html.H6("📖 Scenario Explanations", className="mt-3 mb-2"),
                            html.Ul([
                                html.Li([html.Strong("Best Case (95%): "), "Only 5% of simulations performed better"]),
                                html.Li([html.Strong("Expected (50%): "), "Median outcome - half above, half below"]),
                                html.Li([html.Strong("Worst Case (5%): "), "Only 5% of simulations performed worse"])
                            ], style={"font-size": "0.85rem", "color": "#cccccc"})
                        ], className="explanation")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Risk Assessment"),
                    dbc.CardBody([
                        html.Div([
                            html.H4(f"{monte_carlo['probability_of_loss']:.1%}", className="text-danger mb-2"),
                            html.P("Probability of Loss", className="text-muted mb-3"),
                            
                            html.P([
                                "This represents the likelihood that your strategy will lose money ",
                                "based on historical patterns. A lower percentage indicates a more ",
                                "reliable strategy, while a higher percentage suggests greater risk."
                            ], style={"font-size": "0.9rem"}),
                            
                            html.Div([
                                html.H6("🎯 Risk Interpretation", className="mt-3 mb-2"),
                                html.Ul([
                                    html.Li("< 20%: Low risk strategy"),
                                    html.Li("20-40%: Moderate risk"),
                                    html.Li("> 40%: High risk strategy")
                                ], style={"font-size": "0.85rem", "color": "#cccccc"})
                            ], className="explanation")
                        ])
                    ])
                ])
            ], width=6),
        ])
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