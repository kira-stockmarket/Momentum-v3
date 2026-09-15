import yfinance as yf
import pandas as pd
import os

def get_nifty50_tickers():
    """Scrapes the live Nifty 50 constituents from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/NIFTY_50'
    tables = pd.read_html(url)
    
    # The constituent table is usually the third table on the page
    df = tables[2]
    
    # Extract symbols and append '.NS' for Yahoo Finance NSE formatting
    tickers = df['Symbol'].astype(str) + '.NS'
    return tickers.tolist()

def download_and_save_data():
    os.makedirs('nifty50_data', exist_ok=True)
    tickers = get_nifty50_tickers()
    
    print(f"Starting download for {len(tickers)} Nifty 50 stocks...")
    
    # Download data using multithreading for speed
    # group_by='ticker' ensures we get distinct DataFrames per symbol
    data = yf.download(
        tickers=tickers,
        period="max",
        interval="1d",
        group_by='ticker',
        threads=True,
        auto_adjust=True # Adjusts for stock splits and dividends
    )
    
    for ticker in tickers:
        try:
            # Extract the specific ticker's dataframe
            df = data[ticker].dropna(how='all')
            
            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue
                
            # Downcast to float32 to save 50% RAM during ML training
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
