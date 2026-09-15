import pandas as pd
import lightgbm as lgb
import datetime

DATA_FILE = "dataset.parquet"
PROBA_THRESHOLD = 0.70

def run_daily_scanner():
    print("--- DAILY BREAKOUT SCANNER (TODAY ONLY) ---")
    print(f"Loading market data from {DATA_FILE}...\n")
    
    try:
        df = pd.read_parquet(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
        return

    if 'ticker' not in df.columns:
        print("Error: Could not find 'ticker' column in the dataset.")
        return
        
    # 1. ISOLATE TODAY'S DATA (The absolute last row for each individual stock)
    print("Isolating the most recent parameters for each stock...")
    
    # If your data has a date/time column, sort first to guarantee the last row is the newest
    date_cols = [c for c in df.columns if c.lower() in ['date', 'timestamp', 'datetime']]
    if date_cols:
        df = df.sort_values(date_cols[0])
        
    # Grab only the very last row per ticker
    latest_data = df.groupby('ticker').tail(1).copy()
    
    # The historical dataset used for training excludes those final rows to prevent data leakage
    historical_data = df.drop(latest_data.index)
    
    # 2. SETUP FEATURES AND TARGET
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    if 'target_breakout_20' not in df.columns:
        print("Error: 'target_breakout_20' column missing for training.")
        return
        
    X_train = historical_data[feature_cols]
    y_train = historical_data['target_breakout_20']
    X_latest = latest_data[feature_cols]
    
    # 3. TRAIN MODEL ON HISTORY
    print("Training LightGBM model on historical patterns...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        objective='binary', scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        subsample=0.7, num_leaves=15, n_estimators=200, max_depth=3, learning_rate=0.05, colsample_bytree=0.7
    )
    model.fit(X_train, y_train)
    
    # 4. PREDICT *ONLY* ON TODAY'S PARAMETERS
    print("Evaluating today's live parameters against the model...\n")
    latest_data['breakout_probability'] = model.predict_proba(X_latest)[:, 1]
    
    # Filter strictly for today's setups meeting the 70% threshold
    watchlist = latest_data[latest_data['breakout_probability'] >= PROBA_THRESHOLD].copy()
    watchlist = watchlist.sort_values(by='breakout_probability', ascending=False)
    
    # 5. PRINT RESULTS
    print("=====================================================")
    print(f"🔥 TODAY'S LIVE WATCHLIST ({datetime.date.today()}) 🔥")
    print("=====================================================\n")
    
    if len(watchlist) == 0:
        print("No stocks met the 70% breakout threshold based on today's parameters.")
        print("Cash is a position. Stay patient.")
    else:
        print(f"Found {len(watchlist)} high-conviction setups for today:\n")
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
