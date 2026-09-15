import pandas as pd
import numpy as np
import lightgbm as lgb
import os

DATA_FILE = "dataset.parquet"
PROBA_THRESHOLD = 0.70
TAKE_PROFIT = 0.20
STOP_LOSS = -0.06
TIME_STOP = 21
FRICTION = 0.0015
INITIAL_CAPITAL = 100000.0
MAX_OPEN_POSITIONS = 5

def get_last_10_trades():
    print("Loading dataset for 70%+ conviction trade log extraction...")
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_parquet(DATA_FILE)
    
    if 'date' not in df.columns or 'ticker' not in df.columns:
        print("Error: Dataset must contain both 'date' and 'ticker' columns.")
        return

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'ticker']).reset_index(drop=True)

    # 80/20 chronological split for out-of-sample verification
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X_train, y_train = train_df[feature_cols], train_df['target_breakout_20']
    X_test = test_df[feature_cols]
    
    print("Training LightGBM model to evaluate out-of-sample trades...")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    
    model = lgb.LGBMClassifier(
        objective='binary', scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        subsample=0.7, num_leaves=15, n_estimators=200, max_depth=3, learning_rate=0.05, colsample_bytree=0.7
    )
    model.fit(X_train, y_train)
    
    test_df['predicted_prob'] = model.predict_proba(X_test)[:, 1]
    test_df['signal'] = test_df['predicted_prob'] >= PROBA_THRESHOLD
    
    # Simulation engine state
    available_cash = INITIAL_CAPITAL
    open_positions = []
    trade_logs = []
    
    for row in test_df.itertuples():
        current_idx = row.Index
        current_date = row.date
        current_ticker = row.ticker
        
        # 1. Manage active open positions
        positions_to_remove = []
        for pos in open_positions:
            days_held = current_idx - pos['entry_idx']
            
            hit_sl = getattr(row, 'Low') <= pos['sl_price']
            hit_tp = getattr(row, 'High') >= pos['tp_price']
            time_stop = days_held >= TIME_STOP
            
            exit_price = None
            if hit_sl: exit_price = pos['sl_price']
            elif hit_tp: exit_price = pos['tp_price']
            elif time_stop: exit_price = getattr(row, 'Close')
                
            if exit_price is not None:
                gross_return = (exit_price - pos['entry_price']) / pos['entry_price']
                net_return = gross_return - FRICTION
                trade_pnl = pos['capital_allocated'] * net_return
                available_cash += (pos['capital_allocated'] + trade_pnl)
                
                pos['exit_date'] = current_date.strftime('%Y-%m-%d')
                pos['exit_price'] = exit_price
                pos['return_pct'] = net_return * 100
                trade_logs.append(pos)
                positions_to_remove.append(pos)
                
        for pos in positions_to_remove:
            open_positions.remove(pos)
            
        # 2. Process new signals strictly matching 70%+ probability
        if getattr(row, 'signal') and len(open_positions) < MAX_OPEN_POSITIONS:
            current_open_value = sum([p['capital_allocated'] * (getattr(row, 'Close') / p['entry_price']) for p in open_positions])
            total_equity = available_cash + current_open_value
            position_size = total_equity / MAX_OPEN_POSITIONS
            
            if available_cash >= (position_size * 0.99):
                entry_price = getattr(row, 'Close')
                new_pos = {
                    'ticker': current_ticker,
                    'entry_date': current_date.strftime('%Y-%m-%d'),
                    'entry_price': entry_price,
                    'tp_price': entry_price * (1 + TAKE_PROFIT),
                    'sl_price': entry_price * (1 + STOP_LOSS),
                    'probability': row.predicted_prob * 100,
                    'entry_idx': current_idx,
                    'capital_allocated': position_size
                }
                open_positions.append(new_pos)
                available_cash -= position_size

    # 3. EXTRACT LAST 10 TRADES
    print("\n" + "=" * 90)
    print("🎯 LAST 10 COMPLETED TRADES (70%+ PROBABILITY CONVICTION) 🎯")
    print("=" * 90)
    
    if len(trade_logs) == 0:
        print("No completed trades found meeting the strict 70% threshold.")
        return
        
    last_10 = trade_logs[-10:]
    
    print(f"{'ENTRY DATE':<12} | {'TICKER':<10} | {'PROB':<6} | {'ENTRY (₹)':<10} | {'TARGET (+20%)':<14} | {'SL (-6%)':<10} | {'RETURN':<8}")
    print("-" * 90)
    
    for t in last_10:
        print(f"{t['entry_date']:<12} | {t['ticker']:<10} | {t['probability']:>4.1f}%  | ₹{t['entry_price']:<9.2f} | ₹{t['tp_price']:<13.2f} | ₹{t['sl_price']:<9.2f} | {t['return_pct']:>+.2f}%")
        
    print("=" * 90)

if __name__ == "__main__":
    get_last_10_trades()
