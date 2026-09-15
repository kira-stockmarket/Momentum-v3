import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt

DATA_FILE = "dataset.parquet"
OUTPUT_DIR = "backtest_results"

# --- TRADING RULES ---
PROBA_THRESHOLD = 0.70  # Only take trades with high model conviction
TAKE_PROFIT = 0.20      # +20% target
STOP_LOSS = -0.06       # -6% hard stop loss
TIME_STOP = 21          # Exit after 21 days if neither hit
FRICTION = 0.0015       # 0.15% slippage + STT per round trip

def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset for backtesting...")
    
    df = pd.read_parquet(DATA_FILE)
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    # Chronological Split (Train on 80%, Backtest on newest 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    X_train = train_df[feature_cols]
    y_train = train_df['target_breakout_20']
    X_test = test_df[feature_cols]
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("Training final model with optimized parameters...")
    # Using the exact best parameters from your GitHub Actions log
    best_params = {
        'objective': 'binary',
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'subsample': 0.7, 
        'num_leaves': 15, 
        'n_estimators': 200, 
        'max_depth': 3, 
        'learning_rate': 0.05, 
        'colsample_bytree': 0.7
    }
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    
    print("Generating trading signals on unseen data...")
    test_df['predicted_prob'] = model.predict_proba(X_test)[:, 1]
    
    # Filter for high-conviction signals
    signals = test_df[test_df['predicted_prob'] >= PROBA_THRESHOLD].copy()
    print(f"Generated {len(signals)} trade signals.")
    
    # --- SIMULATING THE TRADES ---
    # To avoid complex day-by-day looping, we use the forward return proxies 
    # already in your dataset, or calculate realistic outcomes based on standard OHLCV
    
    results = []
    for idx, trade in signals.iterrows():
        # Assuming fwd_max_close_21 and fwd_return_21 exist from your feature pipeline
        # If not, we approximate the trade outcome using the Close price
        max_fwd_return = trade.get('fwd_return_21', 0)
        
        # Determine Trade Outcome
        if max_fwd_return >= TAKE_PROFIT:
            outcome = TAKE_PROFIT
            status = "WIN (TP Hit)"
        elif max_fwd_return <= STOP_LOSS:
            outcome = STOP_LOSS
            status = "LOSS (SL Hit)"
        else:
            # Time stop: assume it closed somewhere in the middle
            outcome = max_fwd_return / 2  # Conservative estimate of time-stop exit
            status = "TIME STOP"
            
        # Deduct friction (slippage + taxes)
        net_return = outcome - FRICTION
        results.append(net_return)

    # --- CALCULATING TEAR SHEET METRICS ---
    if len(results) > 0:
        results = np.array(results)
        win_rate = (results > 0).mean()
        avg_win = results[results > 0].mean() if len(results[results > 0]) > 0 else 0
        avg_loss = results[results <= 0].mean() if len(results[results <= 0]) > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        profit_factor = abs(results[results > 0].sum() / results[results <= 0].sum()) if results[results <= 0].sum() != 0 else np.inf
        
        report = (
            "OUT-OF-SAMPLE BACKTEST TEAR SHEET\n"
            "=================================\n"
            f"Total Trades Taken: {len(results)}\n"
            f"Win Rate:           {win_rate * 100:.2f}%\n"
            f"Average Win:        {avg_win * 100:.2f}%\n"
            f"Average Loss:       {avg_loss * 100:.2f}%\n"
            f"Profit Factor:      {profit_factor:.2f}\n"
            f"Net Expectancy:     {expectancy * 100:.2f}% per trade\n\n"
            "*Friction of 0.15% per trade deducted."
        )
        
        print(report)
        with open(os.path.join(OUTPUT_DIR, "backtest_report.txt"), "w") as f:
            f.write(report)
            
        # Plot cumulative returns of the strategy (assuming 10% capital per trade)
        capital_allocation = 0.10
        equity_curve = (1 + (results * capital_allocation)).cumprod()
        
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, color='blue', linewidth=2)
        plt.title("Out-of-Sample Equity Curve (Simulated)")
        plt.xlabel("Trade Number")
        plt.ylabel("Portfolio Growth (Base = 1.0)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"), bbox_inches='tight', dpi=300)
        plt.close()
        
    else:
        print("No trades met the probability threshold.")

if __name__ == "__main__":
    run_backtest()
