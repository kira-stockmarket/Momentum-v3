import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os
import gc
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

DATA_FILE = "dataset.parquet"
OUTPUT_DIR = "ml_insights"

def optimize_memory(df):
    """Downcasts float64 to float32 to cut RAM usage in half."""
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

def train_and_extract_rules():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading dataset from {DATA_FILE}...")
    df = pd.read_parquet(DATA_FILE)
    df = optimize_memory(df)
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    X = df[feature_cols]
    y = df['target_breakout_20']
    
    print(f"Dataset Size: {len(X)} rows | Features: {len(feature_cols)}")
    
    # Chronological Split (Train on older data, Test on newest 20%)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Free up memory immediately
    del df
    gc.collect() 
    
    # Class imbalance weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Positive Class Weighting: {scale_pos_weight:.2f}x")

    # --- 1. MAX UTILIZATION: HYPERPARAMETER TUNING ---
    print("\nStarting Hyperparameter Search (Max CPU Utilization)...")
    
    base_model = lgb.LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1 # Uses all available CPU cores
    )
    
    # Parameter grid for random search
    param_dist = {
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'subsample': [0.7, 0.8, 1.0]
    }
    
    # TimeSeriesSplit prevents future data from leaking into past validation folds
    tscv = TimeSeriesSplit(n_splits=3)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10, 
        scoring='average_precision', # Optimizes for catching rare breakouts
        cv=tscv,
        n_jobs=-1, # Parallel cross-validation
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print(f"Best Parameters Found: {random_search.best_params_}")

    # --- 2. PERFORMANCE METRICS ---
    probs = best_model.predict_proba(X_test)[:, 1]
    preds = best_model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs) # PR-AUC is key for imbalanced data
    
    print("\n--- Out-of-Sample Performance ---")
    print(f"ROC-AUC Score: {roc_auc:.3f}")
    print(f"PR-AUC Score:  {pr_auc:.3f}")
    print(classification_report(y_test, preds, target_names=["Consolidating", "20% Surge"]))

    # --- 3. EXPLAINABLE AI (SHAP) ---
    print("\nGenerating SHAP Explainability rules (Extracting 'Common Things')...")
    
    # Sample actual historical breakouts to find out WHY they broke out
    positive_samples = X[y == 1].sample(n=min(5000, len(X[y==1])), random_state=42)
    
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(positive_samples)
    
    shap_vals_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    # Save the SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals_to_plot, positive_samples, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "breakout_precursors_shap.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save mathematical ranking report
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance (Mean Absolute SHAP)': np.abs(shap_vals_to_plot).mean(axis=0)
    }).sort_values(by='Importance (Mean Absolute SHAP)', ascending=False)
    
    report_path = os.path.join(OUTPUT_DIR, "momentum_rules_report.txt")
    with open(report_path, "w") as f:
        f.write("TOP 15-DAY PRECURSORS TO A 20%+ FRESH BREAKOUT\n")
        f.write("==============================================\n")
        f.write(f"Best Model Params: {random_search.best_params_}\n")
        f.write(f"Out-of-Sample ROC-AUC: {roc_auc:.3f}\n")
        f.write(f"Out-of-Sample PR-AUC: {pr_auc:.3f}\n\n")
        f.write("FEATURES RANKED BY PREDICTIVE IMPACT:\n\n")
        f.write(feature_importance.to_string(index=False))
        
    print(f"\nTraining complete. Insights saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    train_and_extract_rules()
