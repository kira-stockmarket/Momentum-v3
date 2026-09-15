import pandas as pd
import lightgbm as lgb
import datetime

DATA_FILE = "dataset.parquet"
PROBA_THRESHOLD = 0.01

def run_daily_scanner():
    print("--- ACCURATE DAILY BREAKOUT SCANNER ---")
    
    try:
        df = pd.read_parquet(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
        return

    if 'ticker' not in df.columns or 'date' not in df.columns:
        print("Error: Dataset must contain both 'ticker' and 'date' columns.")
        return

    df['date'] = pd.to_datetime(df['date'])
    
    # Find the absolute latest trading date present in the dataset
    latest_dataset_date = df['date'].max()
    print(f"Latest available market date in dataset: {latest_dataset_date.date()}")

    # 1. ISOLATE EXACTLY THE LATEST DATE'S ROWS FOR EACH TICKET
    latest_data = df[df['date'] == latest_dataset_date].copy()
    
    # Historical data is everything strictly before this latest date
    historical_data = df[df['date'] < latest_dataset_date]
    
    if len(latest_data) == 0:
        print("Error: No data found for the latest date.")
        return

    # 2. SETUP FEATURES AND TARGET
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X_train = historical_data[feature_cols]
    y_train = historical_data['target_breakout_20']
    X_latest = latest_data[feature_cols]
    
    # 3. TRAIN PRODUCTION MODEL
    print("Training LightGBM model on historical data...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    
    model = lgb.LGBMClassifier(
        objective='binary', scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        subsample=0.7, num_leaves=15, n_estimators=200, max_depth=3, learning_rate=0.05, colsample_bytree=0.7
    )
    model.fit(X_train, y_train)
    
    # 4. PREDICT ON LATEST DATE
    print("Scanning market parameters for high-conviction setups...\n")
    latest_data['breakout_probability'] = model.predict_proba(X_latest)[:, 1]
    
    # Filter for strict conviction threshold
    watchlist = latest_data[latest_data['breakout_probability'] >= PROBA_THRESHOLD].copy()
    watchlist = watchlist.sort_values(by='breakout_probability', ascending=False)
    
    # 5. PRINT RESULTS
    print("=====================================================")
    print(f"🔥 WATCHLIST FOR DATE: {latest_dataset_date.date()} 🔥")
    print("=====================================================\n")
    
    if len(watchlist) == 0:
        print(f"No stocks met the 70% breakout threshold for {latest_dataset_date.date()}.")
        print("Cash is a position. Stay patient.")
    else:
        print(f"Found {len(watchlist)} high-conviction setups:\n")
        print(f"{'TICKER':<15} | {'CONVICTION (%)':<15} | {'CLOSE (₹)':<15}")
        print("-" * 50)
        for _, row in watchlist.iterrows():
            ticker = row['ticker']
            prob = row['breakout_probability'] * 100
            close_price = row.get('Close', 0.0)
            print(f"{ticker:<15} | {prob:>5.1f}%          | ₹{close_price:<10.2f}")
            
    print("\n=====================================================")
    print("EXECUTION RULES: Enter at 3:25 PM. Hard SL at -6%. Target +20%.")

if __name__ == "__main__":
    run_daily_scanner()
