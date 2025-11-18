import pandas as pd
import backtester
from my_strategies import MovingAverageCrossStrategy
from backtester.sizer import FixedSizeSizer

# Import your library's data fetcher
from enode_quant.api.candles import get_stock_candles

def fetch_data_from_db():
    """
    Fetches 1 year of hourly data for AAPL and GOOGL from the RDS.
    """
    print("--- Fetching Data from RDS ---")
    
    tickers = ['AAPL', 'GOOGL']
    # Adjusted to your date range
    start_date = '2025-01-01'
    end_date = '2025-11-10' 
    resolution = 'H' # Based on your successful test
    
    data_dict = {}

    for symbol in tickers:
        print(f"Fetching {symbol}...")
        try:
            # Use your library to get the DataFrame directly
            df = get_stock_candles(
                symbol=symbol,
                resolution=resolution,
                start_date=start_date,
                end_date=end_date,
                limit=5000, # Ensure we get enough rows
                as_dataframe=True
            )
            
            if df is None or df.empty:
                print(f"WARNING: No data returned for {symbol}")
                continue
                
            # Ensure 'symbol' column exists for the Pydantic model
            # Your query builder likely adds it, but good to be safe
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
    
    # 2. Run the backtest using your simple wrapper
    print("\n--- Starting Backtest ---")
    results = backtester.run_backtest(
        data_dict=data,
        strategy_class=MovingAverageCrossStrategy,
        initial_cash=100_000.0,
        sizer=FixedSizeSizer(10) # Buy 10 shares per signal
    )
    
    # 3. Analyze Results
    print("\n--- Analysis Results ---")
    
    # Get the trade log
    trade_log = backtester.get_trade_log_df(results)
    
    if not trade_log.empty:
        print(f"Total Trades: {len(trade_log)}")
        print("\nRecent Trades:")
        print(trade_log.tail())
        
        # Calculate simple total return
        final_value = results.equity_curve[-1][1]
        pnl = final_value - 100_000
        print(f"\nFinal Portfolio Value: ${final_value:,.2f}")
        print(f"Total P&L: ${pnl:,.2f}")
    else:
        print("No trades were made. Check strategy logic or data.")

    # Optional: Generate full Pyfolio report
    # backtester.generate_full_tear_sheet(results)