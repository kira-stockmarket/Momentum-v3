import pandas as pd
import numpy as np

def build_features():
    print("Loading raw Nifty 50 data...")
    # Adjust file name if your raw file is named differently
    try:
        df = pd.read_parquet("nifty50_data/raw_data.parquet") # Or your raw file source
    except:
        # Fallback if reading from main dataset or csv
        df = pd.read_parquet("dataset.parquet")

    # Ensure date is a formal column, not trapped in the index
    if 'date' not in [c.lower() for c in df.columns]:
        if pd.api.types.is_datetime64_any_dtype(df.index) or 'date' in str(df.index.dtype).lower():
            df = df.reset_index()
        else:
            raise KeyError("Critical: No date column or datetime index found in raw data!")

    # Standardize date column name to lowercase 'date'
    date_col = [c for c in df.columns if c.lower() == 'date'][0]
    df = df.rename(columns={date_col: 'date'})
    df['date'] = pd.to_datetime(df['date'])

    print("Engineering features...")
    # Sort chronologically by ticker and date
    df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)

    # --- FEATURE ENGINEERING LOGIC ---
    # Example momentum & volatility features
    df['daily_return'] = df.groupby('ticker')['Close'].pct_change()
    
    # 15-day Volatility Contraction & Range
    df['feat_price_range_15d'] = (df['High'].rolling(15).max() - df['Low'].rolling(15).min()) / df['Close']
    df['feat_dist_from_15d_high'] = (df['High'].rolling(15).max() - df['Close']) / df['Close']
    df['feat_return_15d'] = df.groupby('ticker')['Close'].pct_change(15)
    
    # Moving Average Distances
    sma50 = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(50).mean())
    sma200 = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(200).mean())
    df['feat_dist_sma50'] = (df['Close'] - sma50) / sma50
    df['feat_dist_sma200'] = (df['Close'] - sma200) / sma200

    # Volatility and volume proxies
    df['feat_volatility_contraction_15_60'] = df['Close'].rolling(15).std() / df['Close'].rolling(60).std()
    df['feat_vol_dryup_15_50'] = df['Volume'].rolling(15).mean() / df['Volume'].rolling(50).mean()
    df['feat_vol_burst_t0'] = df['Volume'] / df['Volume'].rolling(15).mean()
    
    # Up/Down Volume Ratio
    up_vol = np.where(df['daily_return'] > 0, df['Volume'], 0)
    down_vol = np.where(df['daily_return'] < 0, df['Volume'], 0)
    df['feat_up_down_vol_ratio_15'] = pd.Series(up_vol).rolling(15).sum() / (pd.Series(down_vol).rolling(15).sum() + 1e-5)

    # Target: 20% breakout within next 21 days
    fwd_max_high = df.groupby('ticker')['High'].transform(lambda x: x.shift(-1).rolling(21).max())
    df['target_breakout_20'] = (fwd_max_high >= df['Close'] * 1.20).astype(int)

    # Drop NaNs resulting from rolling windows
    df = df.dropna().reset_index(drop=True)

    # Save final dataset with date intact
    df.to_parquet("dataset.parquet")
    print(f"Successfully saved feature-engineered dataset with dates. Total rows: {len(df)}")

if __name__ == "__main__":
    build_features()
