import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = "nifty50_data"
OUTPUT_FILE = "dataset.parquet"

def engineer_features_for_stock(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.sort_index().copy()
    
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()
            
    # Clean zero / missing volumes
    df['Volume'] = df['Volume'].replace(0, np.nan).ffill()

   # --- 1. TARGET LABELING (Look-ahead 21 trading days = ~1 month) ---
    # Find the maximum close in the next 21 days
    indexer_fwd = pd.api.indexers.FixedForwardWindowIndexer(window_size=21)
    df['fwd_max_close_21'] = df['Close'].rolling(window=indexer_fwd).max()
    
    # Calculate the forward maximum return
    df['fwd_return_21'] = (df['fwd_max_close_21'] - df['Close']) / df['Close']
    
    # --- NEW: THE "FRESH BREAKOUT" FILTER ---
    # Find the minimum close in the PAST 21 days
    df['past_min_close_21'] = df['Close'].rolling(window=21).min()
    
    # Calculate how much it has already run up from its recent low
    df['past_runup_21'] = (df['Close'] - df['past_min_close_21']) / df['past_min_close_21']
    
    # It is only a valid target IF it surges >= 20% going forward, 
    # AND it hasn't already run up more than 10% in the recent past (it was consolidating)
    is_massive_surge = df['fwd_return_21'] >= 0.20
    is_consolidating = df['past_runup_21'] <= 0.10
    
    df['target_breakout_20'] = (is_massive_surge & is_consolidating).astype(int)
    
    # Forward max return: (fwd_max - current_close) / current_close
    df['fwd_return_21'] = (df['fwd_max_close_21'] - df['Close']) / df['Close']
    
    # Binary Target: 1 if surge >= 20%, else 0
    df['target_breakout_20'] = (df['fwd_return_21'] >= 0.20).astype(int)

    # --- 2. 15-DAY PRE-BREAKOUT FEATURES (Only past data) ---
    df['daily_return'] = df['Close'].pct_change()

    # Volatility Contraction: 15-day std vs 60-day std
    std_15 = df['daily_return'].rolling(15).std()
    std_60 = df['daily_return'].rolling(60).std()
    df['feat_volatility_contraction_15_60'] = std_15 / (std_60 + 1e-7)

    # Price range compression over last 15 days
    high_15 = df['High'].rolling(15).max()
    low_15 = df['Low'].rolling(15).min()
    df['feat_price_range_15d'] = (high_15 - low_15) / df['Close']

    # Distance from 15-day High (proximity to resistance)
    df['feat_dist_from_15d_high'] = (df['Close'] - high_15) / high_15

    # Volume Indicators
    vol_sma_15 = df['Volume'].rolling(15).mean()
    vol_sma_50 = df['Volume'].rolling(50).mean()
    
    # Volume Dry-up: 15-day average volume vs 50-day average volume
    df['feat_vol_dryup_15_50'] = vol_sma_15 / (vol_sma_50 + 1e-7)

    # Day-0 Volume Burst (Volume today vs 15-day average)
    df['feat_vol_burst_t0'] = df['Volume'] / (vol_sma_15 + 1e-7)

    # Up/Down Volume Ratio over past 15 days
    is_up_day = df['Close'] > df['Open']
    up_vol_15 = (df['Volume'] * is_up_day).rolling(15).sum()
    down_vol_15 = (df['Volume'] * (~is_up_day)).rolling(15).sum()
    df['feat_up_down_vol_ratio_15'] = up_vol_15 / (down_vol_15 + 1e-7)

    # Trend Context: Distance from 50-day and 200-day Simple Moving Average
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    df['feat_dist_sma50'] = (df['Close'] - sma_50) / sma_50
    df['feat_dist_sma200'] = (df['Close'] - sma_200) / sma_200

    # 15-day Cumulative Return
    df['feat_return_15d'] = df['Close'].pct_change(15)

    df['ticker'] = ticker

    # Drop early warm-up rows (200 SMA) and forward unlabelled rows (last 21 days)
    clean_df = df.iloc[200:-21].dropna().copy()
    return clean_df

def build_full_dataset():
    parquet_files = glob.glob(f"{DATA_DIR}/*.parquet")
    print(f"Found {len(parquet_files)} parquet files. Processing...")

    processed_dfs = []
    for filepath in parquet_files:
        ticker = os.path.basename(filepath).replace(".parquet", "")
        df = pd.read_parquet(filepath)
        stock_features = engineer_features_for_stock(df, ticker)
        if not stock_features.empty:
            processed_dfs.append(stock_features)

    if not processed_dfs:
        print("No valid data processed. Check file contents.")
        return

    full_dataset = pd.concat(processed_dfs)
    full_dataset.to_parquet(OUTPUT_FILE, engine="pyarrow", index=True)
    
    total_samples = len(full_dataset)
    breakout_samples = full_dataset['target_breakout_20'].sum()
    breakout_pct = (breakout_samples / total_samples) * 100

    print(f"Dataset generated -> {OUTPUT_FILE}")
    print(f"Total rows: {total_samples}")
    print(f"Breakouts (>=20% in 21 days): {breakout_samples} ({breakout_pct:.2f}%)")

if __name__ == "__main__":
    build_full_dataset()
