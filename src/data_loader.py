# src/data_loader.py
import yfinance as yf
import pandas as pd

def fetch_pair(ticker1, ticker2, start_date, end_date):
    """
    Downloads Adjusted Close prices for two assets and cleans the data.
    """
    print(f"--- Fetching data for {ticker1} and {ticker2} ---")
    
    # Download data for both tickers at once
    data = yf.download([ticker1, ticker2], start=start_date, end=end_date, auto_adjust=False)
    
    # Keep only the 'Adj Close' column (this accounts for dividends/splits)
    df = data['Adj Close']
    
    # FIX: yfinance often returns a MultiIndex (e.g., Price -> Ticker). 
    # We just want the tickers as columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Drop any days where one stock has data but the other doesn't
    # (e.g., holidays in different countries)
    df = df.dropna()
    
    # Ensure the columns are in the order we requested
    df = df[[ticker1, ticker2]]
    
    print(f"Successfully loaded {len(df)} trading days.")
    return df