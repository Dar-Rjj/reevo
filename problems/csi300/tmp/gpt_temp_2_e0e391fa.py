import pandas as pd
import numpy as np
def heuristics_v2(df):
    """
    Calculate price momentum factor based on:
    1. Price change over 10-day window
    2. Momentum over 7-day window
    
    Parameters:
    df (pd.DataFrame): Input dataframe with market data (must include 'close')
    
    Returns:
    pd.Series: Factor values indexed by date
    """
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate 10-day price change (percentage)
    price_change = df['close'].rolling(window=10, min_periods=5).apply(
        lambda x: (x[-1] - x[0]) / x[0] if len(x) >= 5 else np.nan
    )
    
    # Calculate 7-day momentum (average of daily changes)
    daily_returns = df['close'].pct_change()
    momentum = daily_returns.rolling(window=7, min_periods=4).mean()
    
    # Combine factors (equal weighting)
    factor = (price_change + momentum) / 2
    
    return factor
