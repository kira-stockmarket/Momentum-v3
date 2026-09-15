import yfinance as yf
import pandas as pd
import os

def get_nifty50_tickers():
    """Returns a hardcoded list of NIFTY 50 constituent symbols."""
    # NIFTY 50 constituents (Update this list manually if the index changes)
    nifty50_symbols = [
        'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK',
        'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BEL', 'BPCL',
        'BHARTIARTL', 'BRITANNIA', 'CIPLA', 'COALINDIA', 'DRREDDY',
        'EICHERMOT', 'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE',
        'HEROMOTOCO', 'HINDALCO', 'HINDUNILVR', 'ICICIBANK', 'ITC',
        'INDUSINDBK', 'INFY', 'JSWSTEEL', 'KOTAKBANK', 'LT',
        'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 'ONGC',
        'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHRIRAMFIN',
        'SUNPHARMA', 'TATAMOTORS', 'TATASTEEL', 'TCS', 'TATACONSUM',
        'TECHM', 'TITAN', 'TRENT', 'ULTRACEMCO', 'WIPRO'
    ]
    
    # Append '.NS' for Yahoo Finance NSE formatting
    return [f"{symbol}.NS" for symbol in nifty50_symbols]

def download_and_save_data():
    os.makedirs('nifty50_data', exist_ok=True)
    tickers = get_nifty50_tickers()
        
    print(f"Starting download for {len(tickers)} Nifty 50 stocks...")
    
    # Download data using multithreading for speed
    data = yf.download(
        tickers=tickers,
        period="max",
        interval="1d",
        group_by='ticker',
        threads=True,
        auto_adjust=True 
    )
    
    for ticker in tickers:
        try:
            # Extract the specific ticker's dataframe
            df = data[ticker].dropna(how='all')
            
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue
                
            # Downcast to float32 to save RAM
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype('float32')
            
            # Save as highly compressed parquet file
            file_path = f"nifty50_data/{ticker.replace('.NS', '')}.parquet"
            df.to_parquet(file_path, engine='pyarrow', index=True)
            print(f"Saved {ticker} -> {len(df)} days of history.")
            
        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

if __name__ == "__main__":
    download_and_save_data()
