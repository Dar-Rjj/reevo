import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original DataFrame
    df = df.copy()
    
    # Calculate daily price range (normalized)
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    # Calculate 10-day moving average of daily ranges (historical only)
    df['ma_range'] = df['daily_range'].rolling(window=10, min_periods=1).mean()
    
    # Range expansion ratio (current range vs historical average)
    df['range_ratio'] = df['daily_range'] / df['ma_range']
    
    # Volume analysis - calculate percentile rank (0-1) using historical data
    df['vol_rank'] = df['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    
    # Apply logistic scaling to volume rank
    df['vol_score'] = 1 / (1 + np.exp(-10 * (df['vol_rank'] - 0.5)))
    
    # Combine signals - range expansion multiplied by volume confirmation
    factor = df['range_ratio'] * df['vol_score']
    
    return factor
