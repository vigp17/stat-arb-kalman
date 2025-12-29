# src/backtester.py
import numpy as np
import pandas as pd

def backtest_pairs(df, ticker_y, ticker_x, entry_threshold=1.0, stop_loss=2.0):
    """
    Simulates trading with dynamic thresholds.
    
    Parameters:
    - df: DataFrame containing 'Z_Score' and price data
    - ticker_y: The target asset symbol (e.g., 'GDX')
    - ticker_x: The reference asset symbol (e.g., 'GLD')
    - entry_threshold: Z-Score to enter the trade (default 1.0)
    - stop_loss: Z-Score to exit the trade (default 2.0)
    """
    # 1. Initialize logic
    current_position = 0
    signals = []
    
    # 2. Loop through Z-Scores
    for z in df['Z_Score']:
        # STOP LOSS: If spread blows out, get out.
        if abs(z) > stop_loss:
            current_position = 0
            
        # ENTRY LOGIC
        elif z > entry_threshold and current_position != 1: 
            current_position = -1 # Short Spread (Sell Y, Buy X)
        elif z < -entry_threshold and current_position != -1:
            current_position = 1  # Long Spread (Buy Y, Sell X)
            
        # EXIT LOGIC (Mean Reversion)
        elif abs(z) < 0.5:
            current_position = 0
            
        signals.append(current_position)
        
    df['Position'] = signals
    
    # 3. Calculate Returns
    target_ret = df[ticker_y].pct_change()
    ref_ret = df[ticker_x].pct_change()
    
    # Use yesterday's hedge ratio to avoid look-ahead bias
    hedge_ratio = df['beta'].shift(1)
    
    # Strategy Return = Position * (Target_Return - Hedge_Ratio * Reference_Return)
    df['Strategy_Returns'] = df['Position'].shift(1) * (target_ret - hedge_ratio * ref_ret)
    
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod()
    
    return df