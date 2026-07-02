import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Returns using High price
    df['returns'] = df['high'].pct_change()
    
    # Rolling Sum of Returns (15 trading days)
    df['momentum'] = df['returns'].rolling(window=15, min_periods=1).sum()
    
    # Calculate Volume Range (Max Volume - Min Volume) over 15 trading days
    df['volume_range'] = df['volume'].rolling(window=15, min_periods=1).apply(lambda x: x.max() - x.min())
    
    # Scale Momentum by Intraday Volume Impact
    df['factor'] = df['momentum'] / df['volume_range']
    
    # Return the factor as a Series
    return df['factor'].dropna()
