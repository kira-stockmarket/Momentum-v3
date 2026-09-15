import pandas as pd
import lightgbm as lgb
import datetime

DATA_FILE = "dataset.parquet"
PROBA_THRESHOLD = 0.70

def run_daily_scanner():
    print(f"--- DAILY BREAKOUT SCANNER ---")
    print(f"Loading market data from {DATA_FILE}...\n")
    
    try:
        df = pd.read_parquet(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please run your data fetching script first.")
        return

    # 1. Identify the latest data point for each ticker
    if 'ticker' not in df.columns:
        print("Error: Could not find 'ticker' column in the dataset.")
        return
        
    # Sort to ensure the last row is the most recent date for each ticker
    # (Assuming your data fetcher appends newest data at the bottom)
    df = df.sort_index() 
    
    print("Isolating today's closing data for scanning...")
    # Group by ticker and take the absolute last row (latest day) for prediction
    latest_data = df.groupby('ticker').tail(1).copy()
    
    # The rest of the historical data will be used to train the production model
    historical_data = df.drop(latest_data.index)
    
    # 2. Setup Features and Targets
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    if 'target_breakout_20' not in df.columns:
        print("Error: 'target_breakout_20' column missing for training.")
        return
        
    X_train = historical_data[feature_cols]
    y_train = historical_data['target_breakout_20']
    
    X_latest = latest_data[feature_cols]
    
    # 3. Train Production Model
    print("Training production LightGBM model on historical data...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        objective='binary', scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        subsample=0.7, num_leaves=15, n_estimators=200, max_depth=3, learning_rate=0.05, colsample_bytree=0.7
    )
    model.fit(X_train, y_train)
    
    # 4. Generate Live Predictions
    print("Scanning current market structure for tomorrow's breakouts...\n")
    latest_data['breakout_probability'] = model.predict_proba(X_latest)[:, 1]
    
    # Filter for high conviction setups
    watchlist = latest_data[latest_data['breakout_probability'] >= PROBA_THRESHOLD].copy()
    
    # Sort by highest probability
    watchlist = watchlist.sort_values(by='breakout_probability', ascending=False)
    
    # 5. Output the Watchlist
    print("=====================================================")
    print(f"🔥 HIGH CONVICTION WATCHLIST FOR {datetime.date.today() + datetime.timedelta(days=1)} 🔥")
    print("=====================================================\n")
    
    if len(watchlist) == 0:
        print("No stocks met the 70% conviction threshold today.")
        print("Cash is a position. Stay patient.")
    else:
        print(f"Found {len(watchlist)} potential breakouts:\n")
        # Print a formatted table
        print(f"{'TICKER':<15} | {'CONVICTION (%)':<15} | {'LAST CLOSE (₹)':<15}")
        print("-" * 50)
        for _, row in watchlist.iterrows():
            ticker = row['ticker']
            prob = row['breakout_probability'] * 100
            close_price = row.get('Close', 0.0) # Gracefully handle if 'Close' is missing
            print(f"{ticker:<15} | {prob:>5.1f}%          | ₹{close_price:<10.2f}")
            
    print("\n=====================================================")
    print("EXECUTION RULES: Enter at 3:25 PM. Hard SL at -6%. Target +20%.")

if __name__ == "__main__":
    run_daily_scanner()
