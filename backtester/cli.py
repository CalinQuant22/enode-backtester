"""Command-line interface for backtester operations."""

import typer
from pathlib import Path
from typing import Optional
import sys

app = typer.Typer(help="🚀 Enode Backtester CLI - Professional backtesting toolkit")

@app.command()
def dashboard(
    results: str = typer.Argument(..., help="Path to results file (.json or .pkl)"),
    port: int = typer.Option(8050, "--port", "-p", help="Dashboard port"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode"),
):
    """🎯 Launch interactive dashboard for backtest results."""
    
    try:
        from .dashboard import launch_dashboard
        
        if not Path(results).exists():
            typer.echo(f"❌ Results file not found: {results}", err=True)
            raise typer.Exit(1)
        
        typer.echo(f"🚀 Launching dashboard with results from: {results}")
        launch_dashboard(results_path=results, port=port, debug=debug)
        
    except ImportError as e:
        typer.echo(f"❌ Dashboard dependencies missing: {e}", err=True)
        typer.echo("💡 Install with: uv add dash plotly dash-bootstrap-components")
        raise typer.Exit(1)

@app.command()
def run(
    strategy: str = typer.Argument(..., help="Strategy class name"),
    data: str = typer.Option(..., "--data", "-d", help="Data source (file or database)"),
    cash: float = typer.Option(100000, "--cash", "-c", help="Initial cash"),
    output: str = typer.Option("results.json", "--output", "-o", help="Output file"),
    size: int = typer.Option(100, "--size", "-s", help="Position size"),
):
    """🎯 Run backtest with specified parameters."""
    
    typer.echo(f"🔄 Running backtest: {strategy}")
    typer.echo(f"💰 Initial cash: ${cash:,.2f}")
    typer.echo(f"📊 Position size: {size}")
    
    # This would integrate with your existing backtesting logic
    typer.echo("⚠️  Backtest execution not implemented yet")
    typer.echo("💡 Use your existing test.py for now")

@app.command()
def export(
    results: str = typer.Argument(..., help="Path to results file"),
    format: str = typer.Option("csv", "--format", "-f", help="Export format (csv, excel)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """📤 Export backtest results to various formats."""
    
    try:
        from .dashboard.loaders import load_results, export_to_csv, export_equity_curve
        
        portfolio, metrics, monte_carlo = load_results(results)
        
        output_dir = Path(output) if output else Path.cwd()
        output_dir.mkdir(exist_ok=True)
        
        if format == "csv":
            # Export trades
            trades_file = output_dir / "trades.csv"
            export_to_csv(portfolio, str(trades_file))
            
            # Export equity curve
            equity_file = output_dir / "equity_curve.csv"
            export_equity_curve(portfolio, str(equity_file))
            
            typer.echo(f"✅ Exported to {output_dir}")
        
        else:
            typer.echo(f"❌ Unsupported format: {format}")
            raise typer.Exit(1)
            
    except Exception as e:
        typer.echo(f"❌ Export failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def analyze(
    results: str = typer.Argument(..., help="Path to results file"),
    metric: Optional[str] = typer.Option(None, "--metric", "-m", help="Specific metric to display"),
):
    """📊 Analyze backtest results and display key metrics."""
    
    try:
        from .dashboard.loaders import load_results
        
        portfolio, metrics, monte_carlo = load_results(results)
        
        if metric:
            # Display specific metric
            if hasattr(metrics, metric):
                value = getattr(metrics, metric)
                typer.echo(f"{metric}: {value}")
            else:
                typer.echo(f"❌ Unknown metric: {metric}")
                raise typer.Exit(1)
        else:
            # Display summary
            typer.echo("📊 Backtest Results Summary")
            typer.echo("=" * 30)
            typer.echo(f"Total Return:     {metrics.total_return:.2%}")
            typer.echo(f"Sharpe Ratio:     {metrics.sharpe_ratio:.2f}")
            typer.echo(f"Max Drawdown:     {metrics.max_drawdown:.2%}")
            typer.echo(f"Win Rate:         {metrics.win_rate:.1%}")
            typer.echo(f"Profit Factor:    {metrics.profit_factor:.2f}")
            
            if "probability_of_loss" in monte_carlo:
                typer.echo(f"Prob. of Loss:    {monte_carlo['probability_of_loss']:.1%}")
            
    except Exception as e:
        typer.echo(f"❌ Analysis failed: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def compare(
    results1: str = typer.Argument(..., help="First results file"),
    results2: str = typer.Argument(..., help="Second results file"),
):
    """⚖️  Compare two backtest results side by side."""
    
    try:
        from .dashboard.loaders import load_results
        
        _, metrics1, _ = load_results(results1)
        _, metrics2, _ = load_results(results2)
        
        typer.echo("⚖️  Backtest Comparison")
        typer.echo("=" * 50)
        typer.echo(f"{'Metric':<20} {'Result 1':<15} {'Result 2':<15}")
        typer.echo("-" * 50)
        typer.echo(f"{'Total Return':<20} {metrics1.total_return:<15.2%} {metrics2.total_return:<15.2%}")
        typer.echo(f"{'Sharpe Ratio':<20} {metrics1.sharpe_ratio:<15.2f} {metrics2.sharpe_ratio:<15.2f}")
        typer.echo(f"{'Max Drawdown':<20} {metrics1.max_drawdown:<15.2%} {metrics2.max_drawdown:<15.2%}")
        typer.echo(f"{'Win Rate':<20} {metrics1.win_rate:<15.1%} {metrics2.win_rate:<15.1%}")
        
    except Exception as e:
        typer.echo(f"❌ Comparison failed: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()