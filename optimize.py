# optimize.py
from src.data_loader import fetch_pair
from src.kalman import KalmanFilterReg
from src.backtester import backtest_pairs
import numpy as np
import pandas as pd

# 1. Load the "Winner" Data
print("Fetching GDX vs GLD...")
df_raw = fetch_pair('GDX', 'GLD', '2020-01-01', '2024-01-01')

# 2. Run Kalman Filter (Math doesn't change)
kf = KalmanFilterReg(delta=1e-4, R=1e-3)
state_estimates = kf.run_filter(df_raw['GDX'], df_raw['GLD'])
df_raw = df_raw.join(state_estimates)

# Calculate Base Spread
df_raw['Predicted_GDX'] = df_raw['alpha'] + (df_raw['beta'] * df_raw['GLD'])
df_raw['Spread'] = df_raw['GDX'] - df_raw['Predicted_GDX']
df_raw['Z_Score'] = df_raw['Spread'] / df_raw['Spread'].rolling(window=30).std()

# 3. Test different Entry Thresholds
results = []
thresholds = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]

print("\n--- OPTIMIZING ENTRY THRESHOLD ---")

for thresh in thresholds:
    # We copy the data so we don't mess up the original
    df_test = df_raw.copy()
    
    # We assume Stop Loss is always 2x the entry (classic rule of thumb)
    stop_loss = thresh * 2 
    
    # Run a custom backtest logic just for this loop
    # (We duplicate the logic briefly here to inject the variable threshold)
    current_position = 0
    signals = []
    
    for z in df_test['Z_Score']:
        if abs(z) > stop_loss:
            current_position = 0
        elif z > thresh and current_position != 1: 
            current_position = -1
        elif z < -thresh and current_position != -1:
            current_position = 1
        elif abs(z) < 0.5:
            current_position = 0
        signals.append(current_position)
    
    df_test['Position'] = signals
    
    # Calculate Returns
    target_ret = df_test['GDX'].pct_change()
    ref_ret = df_test['GLD'].pct_change()
    hedge_ratio = df_test['beta'].shift(1)
    df_test['Strategy_Returns'] = df_test['Position'].shift(1) * (target_ret - hedge_ratio * ref_ret)
    
    # Stats
    sharpe = (df_test['Strategy_Returns'].mean() / df_test['Strategy_Returns'].std()) * np.sqrt(252)
    total_ret = (1 + df_test['Strategy_Returns'].fillna(0)).cumprod().iloc[-1]
    
    print(f"Threshold {thresh}: Sharpe = {sharpe:.2f} | Return = {total_ret:.2f}x")
    results.append({'Threshold': thresh, 'Sharpe': sharpe, 'Return': total_ret})

# Find best
best = sorted(results, key=lambda x: x['Sharpe'], reverse=True)[0]
print(f"\nWINNER: Threshold {best['Threshold']} with Sharpe {best['Sharpe']:.2f}")