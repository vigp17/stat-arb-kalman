# main.py
from src.data_loader import fetch_pair
from src.kalman import KalmanFilterReg
from src.backtester import backtest_pairs
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
TICKER_Y = 'GDX' # Target (Miners)
TICKER_X = 'GLD' # Reference (Gold)
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

# !!! THE WINNING PARAMETERS (From optimize.py) !!!
ENTRY_THRESHOLD = 1.25
STOP_LOSS = 2.5

def run_analysis():
    print(f"--- Starting Kalman Pairs Trading: {TICKER_Y} vs {TICKER_X} ---")
    
    # 1. Load Data
    df = fetch_pair(TICKER_Y, TICKER_X, START_DATE, END_DATE)

    # 2. Run Kalman Filter
    print(f"Calculating dynamic hedge ratio...")
    kf = KalmanFilterReg(delta=1e-4, R=1e-3)
    state_estimates = kf.run_filter(df[TICKER_Y], df[TICKER_X])
    df = df.join(state_estimates)

    # 3. Calculate Z-Score
    df['Predicted_Y'] = df['alpha'] + (df['beta'] * df[TICKER_X])
    df['Spread'] = df[TICKER_Y] - df['Predicted_Y']
    df['Z_Score'] = df['Spread'] / df['Spread'].rolling(window=30).std()

    # 4. Run Backtest with OPTIMIZED PARAMETERS
    # FIX: We now pass the winning thresholds explicitly!
    df = backtest_pairs(
        df, 
        TICKER_Y, 
        TICKER_X, 
        entry_threshold=ENTRY_THRESHOLD, 
        stop_loss=STOP_LOSS
    )
    
    # 5. Print Results
    final_return = df['Cumulative_Returns'].iloc[-1]
    sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)
    
    print("\n" + "="*30)
    print(f"FINAL PERFORMANCE ({TICKER_Y} vs {TICKER_X})")
    print("="*30)
    print(f"Total Return:    {final_return:.2f}x")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print("="*30)
    
    # 6. Plot Equity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['Cumulative_Returns'], label=f'{TICKER_Y}-{TICKER_X} Strategy', color='green')
    plt.title(f'Kalman Pairs Trading (Threshold={ENTRY_THRESHOLD})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/performance.png')
    print("Chart saved to results/performance.png")
    plt.show()

if __name__ == "__main__":
    run_analysis()