import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Normalize by Open Price
    intraday_momentum = intraday_range / df['open']
    
    # Compute Volume Ratio
    volume_mean = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: x.iloc[:-1].mean())
    volume_ratio = df['volume'] / volume_mean
    
    # Apply Volume Adjustment
    factor = intraday_momentum * volume_ratio
    
    return factor
