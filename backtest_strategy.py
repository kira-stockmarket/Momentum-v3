import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt

DATA_FILE = "dataset.parquet"
OUTPUT_DIR = "backtest_results"

# --- STRICT REAL-WORLD TRADING RULES ---
PROBA_THRESHOLD = 0.70  # Conviction threshold
TAKE_PROFIT = 0.20      # +20% hard target
STOP_LOSS = -0.06       # -6% hard stop loss
TIME_STOP = 21          # 21 trading days max hold
FRICTION = 0.0015       # 0.15% per round trip

def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset for realistic path-dependent backtest...")
    
    # Load and reset index to make row-by-row iteration mathematically simple
    df = pd.read_parquet(DATA_FILE).reset_index(drop=True)
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    X_train, y_train = train_df[feature_cols], train_df['target_breakout_20']
    X_test = test_df[feature_cols]
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("Training final model...")
    model = lgb.LGBMClassifier(
        objective='binary', scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        subsample=0.7, num_leaves=15, n_estimators=200, max_depth=3, learning_rate=0.05, colsample_bytree=0.7
    )
    model.fit(X_train, y_train)
    
    test_df['predicted_prob'] = model.predict_proba(X_test)[:, 1]
    signals = test_df[test_df['predicted_prob'] >= PROBA_THRESHOLD]
    print(f"Generated {len(signals)} high-conviction signals. Running day-by-day simulation...")
    
    # --- REALISTIC TRADE SIMULATION ---
    results = []
    
    for signal_idx in signals.index:
        # Prevent indexing errors at the very end of the dataset
        if signal_idx + 1 >= len(df):
            continue
            
        # 1. NO LOOKAHEAD: We enter on the OPEN of tomorrow (Day t+1)
        entry_price = df.iloc[signal_idx + 1]['Open']
        if pd.isna(entry_price) or entry_price <= 0:
            continue
            
        tp_price = entry_price * (1 + TAKE_PROFIT)
        sl_price = entry_price * (1 + STOP_LOSS)
        
        trade_return = 0
        
        # 2. PATH DEPENDENCY: Step forward day-by-day for a maximum of 21 days
        for step in range(1, TIME_STOP + 1):
            fwd_idx = signal_idx + step
            if fwd_idx >= len(df):
                break
                
            day_data = df.iloc[fwd_idx]
            
            # 3. CONSERVATIVE EXECUTION: If both SL and TP happen on the same day, 
            # assume the worst and record a Stop Loss to prevent false hope.
            if day_data['Low'] <= sl_price:
                trade_return = STOP_LOSS
                break
            elif day_data['High'] >= tp_price:
                trade_return = TAKE_PROFIT
                break
                
            # 4. TIME STOP: If it reaches day 21, exit at the final Close price
            if step == TIME_STOP:
                trade_return = (day_data['Close'] - entry_price) / entry_price
                break
                
        # Deduct slippage and taxes
        results.append(trade_return - FRICTION)

    # --- CALCULATING INSTITUTIONAL METRICS ---
    if len(results) > 0:
        results = np.array(results)
        win_rate = (results > 0).mean()
        avg_win = results[results > 0].mean() if len(results[results > 0]) > 0 else 0
        avg_loss = results[results <= 0].mean() if len(results[results <= 0]) > 0 else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        gross_profits = results[results > 0].sum()
        gross_losses = abs(results[results <= 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else 99.9
        
        report = (
            "REALISTIC PATH-DEPENDENT BACKTEST\n"
            "=================================\n"
            f"Total Trades:       {len(results)}\n"
            f"Win Rate:           {win_rate * 100:.2f}%\n"
            f"Average Win:        {avg_win * 100:.2f}%\n"
            f"Average Loss:       {avg_loss * 100:.2f}%\n"
            f"Profit Factor:      {profit_factor:.2f}\n"
            f"Net Expectancy:     {expectancy * 100:.2f}% per trade\n\n"
            "*Execution on t+1 Open. Conservative SL intra-bar priority applied."
        )
        
        print(report)
        with open(os.path.join(OUTPUT_DIR, "backtest_report.txt"), "w") as f:
            f.write(report)
            
        capital_allocation = 0.10
        equity_curve = (1 + (results * capital_allocation)).cumprod()
        
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, color='blue', linewidth=2)
        plt.title("Out-of-Sample Equity Curve (Simulated)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        print("No trades triggered.")

if __name__ == "__main__":
    run_backtest()
