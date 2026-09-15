import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt

DATA_FILE = "dataset.parquet"
OUTPUT_DIR = "backtest_results"

# --- TRADING RULES ---
PROBA_THRESHOLD = 0.70
TAKE_PROFIT = 0.20
STOP_LOSS = -0.06
TIME_STOP = 21
FRICTION = 0.0015

# --- PORTFOLIO RISK RULES ---
INITIAL_CAPITAL = 100000.0
MAX_OPEN_POSITIONS = 20

def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset for Sequential Event-Driven Backtest...")
    
    # Load data and force a clean sequential index to avoid Date column errors
    df = pd.read_parquet(DATA_FILE).reset_index(drop=True)
    
    # 80/20 Split (Using row order as the timeline proxy)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X_train, y_train = train_df[feature_cols], train_df['target_breakout_20']
    X_test = test_df[feature_cols]
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("Training LightGBM model on the first 80% of historical data...")
    model = lgb.LGBMClassifier(
        objective='binary', scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        subsample=0.7, num_leaves=15, n_estimators=200, max_depth=3, learning_rate=0.05, colsample_bytree=0.7
    )
    model.fit(X_train, y_train)
    
    print(f"Running simulation on the remaining {len(test_df)} out-of-sample rows...")
    test_df['predicted_prob'] = model.predict_proba(X_test)[:, 1]
    test_df['signal'] = test_df['predicted_prob'] >= PROBA_THRESHOLD
    
    # --- EVENT-DRIVEN SIMULATION STATE ---
    available_cash = INITIAL_CAPITAL
    open_positions = []
    equity_curve_values = []
    trade_returns = []
    
    # Iterate row-by-row to simulate the passage of time
    for row in test_df.itertuples():
        current_idx = row.Index
        
        # 1. Manage Existing Open Positions
        positions_to_remove = []
        for pos in open_positions:
            days_held = current_idx - pos['entry_idx']
            
            # SL checked first for conservative risk modeling
            hit_sl = getattr(row, 'Low') <= pos['sl_price']
            hit_tp = getattr(row, 'High') >= pos['tp_price']
            time_stop = days_held >= TIME_STOP
            
            exit_price = None
            if hit_sl:
                exit_price = pos['sl_price']
            elif hit_tp:
                exit_price = pos['tp_price']
            elif time_stop:
                exit_price = getattr(row, 'Close')
                
            if exit_price is not None:
                gross_return = (exit_price - pos['entry_price']) / pos['entry_price']
                net_return = gross_return - FRICTION
                
                # Free up capital and record PnL
                trade_pnl = pos['capital_allocated'] * net_return
                available_cash += (pos['capital_allocated'] + trade_pnl)
                
                trade_returns.append(net_return)
                positions_to_remove.append(pos)
        
        for pos in positions_to_remove:
            open_positions.remove(pos)
            
        # 2. Process New Signals (Only if capacity allows)
        if getattr(row, 'signal') and len(open_positions) < MAX_OPEN_POSITIONS:
            current_open_value = sum([p['capital_allocated'] * (getattr(row, 'Close') / p['entry_price']) for p in open_positions])
            total_equity = available_cash + current_open_value
            position_size = total_equity / MAX_OPEN_POSITIONS
            
            # Ensure physical cash is available
            if available_cash >= (position_size * 0.99):
                entry_price = getattr(row, 'Close')
                open_positions.append({
                    'entry_idx': current_idx, 
                    'entry_price': entry_price, 
                    'capital_allocated': position_size,
                    'sl_price': entry_price * (1 + STOP_LOSS), 
                    'tp_price': entry_price * (1 + TAKE_PROFIT)
                })
                available_cash -= position_size
                    
        # 3. Mark-to-Market Daily Equity
        current_open_value = sum([p['capital_allocated'] * (getattr(row, 'Close') / p['entry_price']) for p in open_positions])
        equity_curve_values.append(available_cash + current_open_value)

    # --- FINAL METRICS ---
    if len(trade_returns) > 0:
        trade_returns = np.array(trade_returns)
        final_cap = equity_curve_values[-1]
        
        win_rate = (trade_returns > 0).mean()
        avg_win = trade_returns[trade_returns > 0].mean() if len(trade_returns[trade_returns > 0]) > 0 else 0
        avg_loss = trade_returns[trade_returns <= 0].mean() if len(trade_returns[trade_returns <= 0]) > 0 else 0
        
        gross_profits = trade_returns[trade_returns > 0].sum()
        gross_losses = abs(trade_returns[trade_returns <= 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
        
        report = (
            "80/20 EVENT-DRIVEN PORTFOLIO TEAR SHEET\n"
            "=================================\n"
            f"Initial Capital:    ₹{INITIAL_CAPITAL:,.2f}\n"
            f"Final Capital:      ₹{final_cap:,.2f}\n"
            f"Total Return:       {((final_cap - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%\n"
            f"Max Active Trades:  {MAX_OPEN_POSITIONS} (20% Equity per trade)\n"
            "---------------------------------\n"
            f"Total Trades Taken: {len(trade_returns)}\n"
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
        plt.plot(equity_curve_values, color='royalblue', linewidth=2)
        plt.title("Realistic Event-Driven Equity Curve (80/20 Split)")
        plt.xlabel("Simulated Timeline (Row Index)")
        plt.ylabel("Portfolio Value (₹)")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        print("No trades executed.")

if __name__ == "__main__":
    run_backtest()
