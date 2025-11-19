"""Main Dash application server."""

import dash
from dash import html, dcc
import plotly.io as pio

from .layout import create_layout
from .callbacks import register_callbacks
from .loaders import load_results

# Set default plotly theme
pio.templates.default = "plotly_dark"

def create_app(results_path: str = None, portfolio=None, metrics=None, monte_carlo=None):
    """Create and configure Dash app."""
    
    import os
    assets_path = os.path.join(os.path.dirname(__file__), 'assets')
    
    app = dash.Dash(__name__, 
                    assets_folder=assets_path,
                    external_stylesheets=[
        "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ])
    
    app.title = "Strategy Performance Dashboard"
    
    # Custom CSS for dark theme
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    background-color: #1a1a1a;
                    color: #ffffff !important;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                
                /* Card styling */
                .card {
                    background-color: #2d2d2d !important;
                    border: 1px solid #404040 !important;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                    color: #ffffff !important;
                }
                .card-body {
                    color: #ffffff !important;
                }
                .card-header {
                    background-color: #3d3d3d !important;
                    border-bottom: 1px solid #404040 !important;
                    font-weight: 600;
                    color: #ffffff !important;
                }
                
                /* Tab styling */
                .nav-tabs .nav-link {
                    background-color: #2d2d2d !important;
                    border: 1px solid #404040 !important;
                    color: #cccccc !important;
                }
                .nav-tabs .nav-link.active {
                    background-color: #0d6efd !important;
                    border-color: #0d6efd !important;
                    color: #ffffff !important;
                }
                
                /* Table styling */
                .table {
                    color: #ffffff !important;
                    background-color: #2d2d2d !important;
                }
                .table td, .table th {
                    color: #ffffff !important;
                    border-color: #404040 !important;
                }
                .table-striped tbody tr:nth-of-type(odd) {
                    background-color: #3d3d3d !important;
                }
                .table-dark {
                    background-color: #2d2d2d !important;
                    color: #ffffff !important;
                }
                
                /* Text colors */
                .text-muted {
                    color: #aaaaaa !important;
                }
                p, h1, h2, h3, h4, h5, h6, span, div {
                    color: inherit;
                }
                
                /* Explanation boxes */
                .explanation {
                    background-color: #2a2a2a !important;
                    border-left: 4px solid #0d6efd;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 8px;
                    font-size: 14px;
                    line-height: 1.5;
                    color: #ffffff !important;
                }
                
                /* Metric cards */
                .metric-card {
                    background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%) !important;
                    border: none !important;
                    border-radius: 15px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
                    color: #ffffff !important;
                }
                .metric-value {
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin-bottom: 5px;
                }
                .metric-label {
                    font-size: 0.9rem;
                    opacity: 0.8;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    color: #cccccc !important;
                }
                
                /* List styling */
                ul, li {
                    color: #ffffff !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    # Load results from path or use direct objects
    if portfolio and metrics and monte_carlo:
        # Direct objects provided - no JSON needed
        app.layout = create_layout(portfolio, metrics, monte_carlo)
    elif results_path:
        # Load from JSON file
        portfolio, metrics, monte_carlo = load_results(results_path)
        app.layout = create_layout(portfolio, metrics, monte_carlo)
    else:
        app.layout = html.Div([
            html.H1("Strategy Performance Dashboard"),
            html.P("No results loaded. Use CLI to load backtest results.")
        ])
    
    # Register callbacks
    register_callbacks(app)
    
    return app

def launch_dashboard(portfolio=None, results_path: str = None, port: int = 8050, debug: bool = False):
    """Launch dashboard server."""
    
    if portfolio and not results_path:
        # Use direct objects - no JSON needed!
        from ..metrics import analyze_strategy
        metrics, monte_carlo = analyze_strategy(portfolio)
        app = create_app(portfolio=portfolio, metrics=metrics, monte_carlo=monte_carlo)
    else:
        # Load from JSON file
        app = create_app(results_path=results_path)
    
    print(f"🚀 Dashboard starting at http://localhost:{port}")
    app.run(debug=debug, port=port, host="0.0.0.0")