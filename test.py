import pandas as pd
import backtester
from my_strategies import MovingAverageCrossStrategy
from backtester.sizer import FixedSizeSizer
from backtester import RiskManager, MaxPositionSizeRule, MaxCashUsageRule, MaxPositionCountRule, analyze_strategy

# Import your library's data fetcher
from enode_quant.api.candles import get_stock_candles

def fetch_data_from_db():
    """
    Fetches 1 year of hourly data for AAPL and GOOGL from the RDS.
    """
    print("--- Fetching Data from RDS ---")
    
    tickers = ['AAPL', 'GOOGL']
    start_date = '2025-01-01'
    end_date = '2025-11-10' 
    resolution = 'H'
    
    data_dict = {}

    for symbol in tickers:
        print(f"Fetching {symbol}...")
        try:
            df = get_stock_candles(
                symbol=symbol,
                resolution=resolution,
                start_date=start_date,
                end_date=end_date,
                limit=5000,
                as_dataframe=True
            )
            
            if df is None or df.empty:
                print(f"WARNING: No data returned for {symbol}")
                continue
                
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            data_dict[symbol] = df
            print(f"  -> Loaded {len(df)} rows for {symbol}")
            
        except Exception as e:
            print(f"ERROR fetching {symbol}: {e}")

    return data_dict

if __name__ == "__main__":
    
    # 1. Get the real data
    data = fetch_data_from_db()
    
    if not data:
        print("CRITICAL: No data loaded. Exiting.")
        exit()
    
    # 2. Run backtest WITHOUT risk management
    print("\n--- Backtest WITHOUT Risk Management ---")
    results_no_risk = backtester.run_backtest(
        data_dict=data,
        strategy_class=MovingAverageCrossStrategy,
        initial_cash=100_000.0,
        sizer=FixedSizeSizer(100)  # Larger size to show risk impact
    )
    
    # 3. Run backtest WITH risk management
    print("\n--- Backtest WITH Risk Management ---")
    risk_manager = RiskManager([
        MaxPositionSizeRule(max_position_pct=0.15),  # Max 15% per position
        MaxCashUsageRule(reserve_cash=5000.0),       # Keep $5k reserve
        MaxPositionCountRule(max_positions=5)        # Max 5 positions
    ])
    
    results_with_risk = backtester.run_backtest(
        data_dict=data,
        strategy_class=MovingAverageCrossStrategy,
        initial_cash=100_000.0,
        sizer=FixedSizeSizer(100),
        risk_manager=risk_manager
    )
    
    # 4. Compare Results
    print("\n=== COMPARISON ===")
    
    trade_log_no_risk = backtester.get_trade_log_df(results_no_risk)
    trade_log_with_risk = backtester.get_trade_log_df(results_with_risk)
    
    final_no_risk = results_no_risk.equity_curve[-1][1]
    final_with_risk = results_with_risk.equity_curve[-1][1]
    
    print(f"WITHOUT risk: ${final_no_risk:,.2f} | {len(trade_log_no_risk)} trades | Cash: ${results_no_risk.current_cash:,.2f}")
    print(f"WITH risk:    ${final_with_risk:,.2f} | {len(trade_log_with_risk)} trades | Cash: ${results_with_risk.current_cash:,.2f}")
    
    pnl_no_risk = final_no_risk - 100_000
    pnl_with_risk = final_with_risk - 100_000
    
    print(f"\nP&L WITHOUT risk: ${pnl_no_risk:,.2f} ({pnl_no_risk/100_000*100:.1f}%)")
    print(f"P&L WITH risk:    ${pnl_with_risk:,.2f} ({pnl_with_risk/100_000*100:.1f}%)")
    
    # 5. Advanced Metrics Analysis
    print("\n--- Advanced Metrics Analysis ---")
    metrics, monte_carlo = analyze_strategy(results_with_risk)
    
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f} (CI: [{metrics.sharpe_confidence_interval[0]:.2f}, {metrics.sharpe_confidence_interval[1]:.2f}])")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"VaR (95%): {metrics.var_95:.2%}")
    
    if "probability_of_loss" in monte_carlo:
        print(f"\nMonte Carlo Analysis:")
        print(f"Probability of Loss: {monte_carlo['probability_of_loss']:.1%}")
        print(f"Expected Range (25%-75%): ${monte_carlo['final_value_percentiles']['25%']:,.0f} - ${monte_carlo['final_value_percentiles']['75%']:,.0f}")
    
    if not trade_log_with_risk.empty:
        print("\nRisk-managed trades:")
        print(trade_log_with_risk.tail())
    
    print("\n✅ Comprehensive backtest completed!")
    # 6. Save Results for Dashboard
    print("\n--- Saving Results ---")
    from backtester.dashboard.loaders import save_results
    
    # Save as JSON (lightweight, shareable)
    save_results(results_with_risk, metrics, monte_carlo, "backtest_results.json")
    print("✅ Results saved to backtest_results.json")
    
    print("\n🚀 Launch dashboard with:")
    print("   uv run python -m backtester.cli dashboard backtest_results.json")
    print("   Or: uv run python -m backtester.cli analyze backtest_results.json")