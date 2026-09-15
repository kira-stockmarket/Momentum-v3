import yfinance as yf
import pandas as pd
import requests
import os
from io import StringIO

def get_nifty50_tickers():
    """Scrapes the live Nifty 50 constituents from Wikipedia using a custom User-Agent."""
    url = 'https://en.wikipedia.org/wiki/NIFTY_50'
    
    # Define a custom User-Agent to bypass Wikipedia's 403 Forbidden error
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Fetch the page content
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Raise an exception if the request fails
    
    # Use StringIO to wrap the HTML string, avoiding pandas deprecation warnings
    html_data = StringIO(response.text)
    
    # Read the tables from the HTML
    tables = pd.read_html(html_data)
    
    # The constituent table is usually the third table on the page
    df = tables[2]
    
    # Extract symbols and append '.NS' for Yahoo Finance NSE formatting
    tickers = df['Symbol'].astype(str) + '.NS'
    return tickers.tolist()

def download_and_save_data():
    os.makedirs('nifty50_data', exist_ok=True)
    
    try:
        tickers = get_nifty50_tickers()
    except Exception as e:
        print(f"Failed to fetch Nifty 50 tickers: {e}")
        return
        
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
