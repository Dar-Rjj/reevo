import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-to-Low Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Normalize Intraday Range by Open Price
    df['normalized_range'] = df['intraday_range'] / df['open']
    
    # Compute 10-day rolling percentile rank of Volume
    df['volume_percentile'] = df['volume'].rolling(window=10).apply(lambda x: (x.rank(pct=True).iloc[-1]))
    
    # Multiply Normalized Range by Volume Percentile Rank
    df['intraday_momentum'] = df['normalized_range'] * df['volume_percentile']
    
    # Calculate Rolling 3-Day Return
    df['daily_return'] = df['close'].pct_change()
    df['rolling_3day_return'] = df['daily_return'].rolling(window=3).sum()
    
    # Weighted Reversal Score
    df['reversal_score'] = df['intraday_momentum'] * df['rolling_3day_return']
    
    # Use Sign to Confirm Reversal Direction
    df['factor'] = np.sign(df['reversal_score'])
    
    return df['factor']
