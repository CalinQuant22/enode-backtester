"""Data persistence for backtest results."""

import json
import pickle
from pathlib import Path
from typing import Tuple, Any

def save_results(portfolio, metrics, monte_carlo, filepath: str):
    """Save backtest results to file."""
    
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        # Save as JSON (limited by serialization)
        data = {
            'metrics': {
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'calmar_ratio': metrics.calmar_ratio,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'skewness': metrics.skewness,
                'kurtosis': metrics.kurtosis,
            },
            'monte_carlo': monte_carlo,
            'equity_curve': [
                {'timestamp': str(ts), 'equity': float(eq)} 
                for ts, eq in portfolio.equity_curve
            ],
            'trade_count': len(portfolio.trade_log),
            'final_cash': portfolio.current_cash,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    elif filepath.suffix == '.pkl':
        # Save as pickle (full object preservation)
        data = {
            'portfolio': portfolio,
            'metrics': metrics,
            'monte_carlo': monte_carlo
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

def load_results(filepath: str) -> Tuple[Any, Any, Any]:
    """Load backtest results from file."""
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create mock portfolio object for JSON data
        portfolio = MockPortfolio(data)
        
        # Create mock metrics object
        metrics = MockMetrics(data['metrics'])
        
        monte_carlo = data['monte_carlo']
        
        return portfolio, metrics, monte_carlo
    
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data['portfolio'], data['metrics'], data['monte_carlo']
    
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

class MockPortfolio:
    """Mock portfolio object for JSON-loaded data."""
    
    def __init__(self, data):
        self.equity_curve = [
            (item['timestamp'], item['equity']) 
            for item in data['equity_curve']
        ]
        self.trade_log = []  # Simplified for JSON
        self.current_cash = data.get('final_cash', 0)

class MockMetrics:
    """Mock metrics object for JSON-loaded data."""
    
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, value)
        
        # Add missing attributes with defaults
        self.return_confidence_interval = (0.0, 0.0)
        self.sharpe_confidence_interval = (0.0, 0.0)

def export_to_csv(portfolio, filepath: str):
    """Export trade log to CSV."""
    
    import pandas as pd
    
    if not portfolio.trade_log:
        print("No trades to export")
        return
    
    trades_data = []
    for trade in portfolio.trade_log:
        trades_data.append({
            'timestamp': trade.timestamp,
            'symbol': trade.symbol,
            'quantity': trade.quantity,
            'price': trade.fill_price,
            'signal': trade.signal,
            'commission': getattr(trade, 'commission', 0)
        })
    
    df = pd.DataFrame(trades_data)
    df.to_csv(filepath, index=False)
    print(f"Trades exported to {filepath}")

def export_equity_curve(portfolio, filepath: str):
    """Export equity curve to CSV."""
    
    import pandas as pd
    
    equity_df = pd.DataFrame(portfolio.equity_curve, columns=['timestamp', 'equity'])
    equity_df.to_csv(filepath, index=False)
    print(f"Equity curve exported to {filepath}")