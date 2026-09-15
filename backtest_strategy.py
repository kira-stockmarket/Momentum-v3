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
MAX_OPEN_POSITIONS = 5

def run_backtest():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading dataset for 5-Year Event-Driven Portfolio Backtest...")
    
    df = pd.read_parquet(DATA_FILE)
    
    # 1. SMART TIMELINE SETUP: Auto-detect the date column
    possible_date_cols = ['date', 'Date', 'timestamp', 'datetime', 'Date/Time', 'time']
    date_col = None
    for col in possible_date_cols:
        if col in df.columns:
            date_col = col
            break
            
    if date_col is None:
        print(f"ERROR: Available columns are: {list(df.columns)}")
        raise KeyError("Could not find a date column in the dataset! Please check the column names printed above.")
        
    print(f"Detected timeline column: '{date_col}'")
    
    # Convert and sort chronologically
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # 2. STRICT 5-YEAR OUT-OF-SAMPLE SPLIT
    latest_date = df[date_col].max()
    cutoff_date = latest_date - pd.DateOffset(years=5)
    
    train_df = df[df[date_col] < cutoff_date]
    test_df = df[df[date_col] >= cutoff_date].copy().reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X_train, y_train = train_df[feature_cols], train_df['target_breakout_20']
    X_test = test_df[feature_cols]
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print(f"Training model on data prior to {cutoff_date.date()}...")
    model = lgb.LGBMClassifier(
        objective='binary', scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        subsample=0.7, num_leaves=15, n_estimators=200, max_depth=3, learning_rate=0.05, colsample_bytree=0.7
    )
    model.fit(X_train, y_train)
    
    print(f"Running simulation on {len(test_df)} rows from {cutoff_date.date()} to {latest_date.date()}...")
    test_df['predicted_prob'] = model.predict_proba(X_test)[:, 1]
    test_df['signal'] = test_df['predicted_prob'] >= PROBA_THRESHOLD
    
    # --- EVENT-DRIVEN SIMULATION STATE ---
    available_cash = INITIAL_CAPITAL
    open_positions = []
    
    # For plotting
    equity_curve_dates = []
    equity_curve_values = []
    trade_returns = []
    
    for row in test_df.itertuples():
        current_idx = row.Index
        current_date = getattr(row, date_col)
        
        # 1. Manage Existing Open Positions
        positions_to_remove = []
        for pos in open_positions:
            days_held = current_idx - pos['entry_idx']
            
            # Check stops and targets (SL checked first)
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
                
                trade_pnl = pos['capital_allocated'] * net_return
                available_cash += (pos['capital_allocated'] + trade_pnl)
                
                trade_returns.append(net_return)
                positions_to_remove.append(pos)
        
        for pos in positions_to_remove:
            open_positions.remove(pos)
            
        # 2. Process New Signals (Entry at Close)
        if getattr(row, 'signal'):
            if len(open_positions) < MAX_OPEN_POSITIONS:
                current_open_value = sum([p['capital_allocated'] * (getattr(row, 'Close') / p['entry_price']) for p in open_positions])
                total_equity = available_cash + current_open_value
                position_size = total_equity / MAX_OPEN_POSITIONS
                
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
                    
        # 3. Record Daily Portfolio Equity
        current_open_value = sum([p['capital_allocated'] * (getattr(row, 'Close') / p['entry_price']) for p in open_positions])
        daily_equity = available_cash + current_open_value
        
        # Only append to curve if the date changed
        if len(equity_curve_dates) == 0 or equity_curve_dates[-1] != current_date:
            equity_curve_dates.append(current_date)
            equity_curve_values.append(daily_equity)

    # --- FINAL METRICS ---
    if len(trade_returns) > 0:
        trade_returns = np.array(trade_returns)
        final_capital = equity_curve_values[-1]
        total_return_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        win_rate = (trade_returns > 0).mean()
        avg_win = trade_returns[trade_returns > 0].mean() if len(trade_returns[trade_returns > 0]) > 0 else 0
        avg_loss = trade_returns[trade_returns <= 0].mean() if len(trade_returns[trade_returns <= 0]) > 0 else 0
        
        gross_profits = trade_returns[trade_returns > 0].sum()
        gross_losses = abs(trade_returns[trade_returns <= 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
        
        report = (
            "5-YEAR EVENT-DRIVEN PORTFOLIO TEAR SHEET\n"
            "=================================\n"
            f"Period:             {cutoff_date.date()} to {latest_date.date()}\n"
            f"Initial Capital:    ₹{INITIAL_CAPITAL:,.2f}\n"
            f"Final Capital:      ₹{final_capital:,.2f}\n"
            f"Total Return:       {total_return_pct:.2f}%\n"
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
            
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve_dates, equity_curve_values, color='darkorange', linewidth=2)
        plt.title(f"5-Year Equity Curve (₹100,000 Initial Capital)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value (₹)")
        plt.grid(True, alpha=0.3)
        plt.gcf().autofmt_xdate()
        plt.savefig(os.path.join(OUTPUT_DIR, "equity_curve.png"), bbox_inches='tight', dpi=300)
        plt.close()
    else:
        print("No trades executed in the last 5 years.")

if __name__ == "__main__":
    run_backtest()
