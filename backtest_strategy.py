import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt

DATA_FILE = "dataset.parquet"
OUTPUT_DIR = "backtest_results"

# --- TRADING RULES ---
PROBA_THRESHOLD = 0.70  # Conviction threshold
TAKE_PROFIT = 0.20      # +20% hard target
STOP_LOSS = -0.06       # -6% hard stop loss
TIME_STOP = 21          # 21 trading days max hold
FRICTION = 0.0015       # 0.15% friction (STT, broker, and 3:29 PM slippage)

# --- PORTFOLIO RISK RULES ---
INITIAL_CAPITAL = 100000.0
MAX_OPEN_POSITIONS = 5  # Max 5 positions means 20% equity allocation per trade

def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset for Portfolio Backtest...")
    
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
    print(f"Generated {len(signals)} signals. Running day-by-day path dependency...")
    
    # --- REALISTIC TRADE SIMULATION (ENTRY AT CLOSE) ---
    trade_returns = []
    
    for signal_idx in signals.index:
        # EXECUTION: Enter at the exact Close of Day t
        entry_price = df.iloc[signal_idx]['Close']
        if pd.isna(entry_price) or entry_price <= 0:
            continue
            
        tp_price = entry_price * (1 + TAKE_PROFIT)
        sl_price = entry_price * (1 + STOP_LOSS)
        
        trade_return = 0
        
        # WALK FORWARD: Day t+1 up to Day 21
        for step in range(1, TIME_STOP + 1):
            fwd_idx = signal_idx + step
            if fwd_idx >= len(df):
                break
                
            day_data = df.iloc[fwd_idx]
            
            # SL Priority checks
            if day_data['Low'] <= sl_price:
                trade_return = STOP_LOSS
                break
            elif day_data['High'] >= tp_price:
                trade_return = TAKE_PROFIT
                break
                
            # Time Stop
            if step == TIME_STOP:
                trade_return = (day_data['Close'] - entry_price) / entry_price
                break
                
        trade_returns.append(trade_return - FRICTION)

    # --- PORTFOLIO SIMULATION & METRICS ---
    if len(trade_returns) > 0:
        trade_returns = np.array(trade_returns)
        
        # Capital Allocation Simulation
        capital = INITIAL_CAPITAL
        position_size_pct = 1.0 / MAX_OPEN_POSITIONS
        equity_curve = [capital]
        
        for r in trade_returns:
            # We risk 20% of the CURRENT compounding capital on each trade
            profit_loss = capital * position_size_pct * r
            capital += profit_loss
            equity_curve.append(capital)
            
        final_capital = equity_curve[-1]
        total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        win_rate = (trade_returns > 0).mean()
        avg_win = trade_returns[trade_returns > 0].mean() if len(trade_returns[trade_returns > 0]) > 0 else 0
        avg_loss = trade_returns[trade_returns <= 0].mean() if len(trade_returns[trade_returns <= 0]) > 0 else 0
        
        gross_profits = trade_returns[trade_returns > 0].sum()
        gross_losses = abs(trade_returns[trade_returns <= 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
        
        report = (
            "PORTFOLIO BACKTEST TEAR SHEET\n"
            "=================================\n"
            f"Initial Capital:    ₹{INITIAL_CAPITAL:,.2f}\n"
            f"Final Capital:      ₹{final_capital:,.2f}\n"
            f"Total Return:       {total_return_pct:.2f}%\n"
            f"Max Active Trades:  {MAX_OPEN_POSITIONS} (20% Equity per trade)\n"
            "---------------------------------\n"
            f"Total Trades:       {len(trade_returns)}\n"
            f"Win Rate:           {win_rate * 100:.2f}%\n"
            f"Average Win:        {avg_win * 100:.2f}%\n"
            f"Average Loss:       {avg_loss * 100:.2f}%\n"
            f"Profit Factor:      {profit_factor:.2f}\n"
            f"Net Expectancy:     {((win_rate * avg_win) + ((1 - win_rate) * avg_loss)) * 100:.2f}% per trade\n"
        )
        
        print(report)
        with open(os.path.join(OUTPUT_DIR, "backtest_report.txt"), "w") as f:
            f.write(report)
            
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, color='green', linewidth=2)
        plt.title("Portfolio Equity Curve (₹100,000 Initial Capital)")
        plt.xlabel("Trade Count")
        plt.ylabel("Portfolio Value (₹)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        print("No trades triggered.")

if __name__ == "__main__":
    run_backtest()
