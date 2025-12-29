# batch_run.py
from src.data_loader import fetch_pair
from src.kalman import KalmanFilterReg
from src.backtester import backtest_pairs
import numpy as np
import pandas as pd

# The "Universe" we want to test
pairs_to_test = [
    ('GDX', 'GLD'),   # Gold Miners vs Gold
    ('EWA', 'EWC'),   # Australia vs Canada (Commodities)
    ('XOM', 'CVX'),   # Exxon vs Chevron (Oil Giants)
    ('KO', 'PEP'),    # Coke vs Pepsi (Beverages)
    ('MS', 'GS'),     # Morgan Stanley vs Goldman Sachs (Banking)
]

results = []

print("--- STARTING BATCH RESEARCH ---")

for ticker_y, ticker_x in pairs_to_test:
    try:
        print(f"\nTesting {ticker_y} vs {ticker_x}...")
        
        # 1. Fetch Data
        df = fetch_pair(ticker_y, ticker_x, '2020-01-01', '2024-01-01')
        
        if df.empty:
            print("No data found, skipping.")
            continue

        # 2. Kalman Filter
        kf = KalmanFilterReg(delta=1e-4, R=1e-3)
        state_estimates = kf.run_filter(df[ticker_y], df[ticker_x])
        df = df.join(state_estimates)

        # 3. Signals
        df['Predicted_Y'] = df['alpha'] + (df['beta'] * df[ticker_x])
        df['Spread'] = df[ticker_y] - df['Predicted_Y']
        df['Z_Score'] = df['Spread'] / df['Spread'].rolling(window=30).std()

        # 4. Backtest
        df = backtest_pairs(df, ticker_y, ticker_x)
        
        # 5. Record Stats
        final_ret = df['Cumulative_Returns'].iloc[-1]
        sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)
        
        results.append({
            'Pair': f"{ticker_y} vs {ticker_x}",
            'Total_Return': final_ret,
            'Sharpe_Ratio': sharpe
        })
        
    except Exception as e:
        print(f"Error testing {ticker_y}/{ticker_x}: {e}")

# --- PRINT LEADERBOARD ---
print("\n" + "="*40)
print("STRATEGY LEADERBOARD")
print("="*40)
results_df = pd.DataFrame(results).sort_values(by='Sharpe_Ratio', ascending=False)
print(results_df)
print("="*40)