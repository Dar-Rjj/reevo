import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Order Flow Asymmetry
    # Calculate rolling sums for high and low
    high_rolling_sum = df['high'].rolling(window=5, min_periods=1).sum()
    low_rolling_sum = df['low'].rolling(window=5, min_periods=1).sum()
    
    # Calculate ratio of rolling sums
    ratio = high_rolling_sum / low_rolling_sum
    
    # Normalize using cross-sectional rank
    normalized_ratio = ratio.rank(pct=True)
    
    # Volume Confirmation
    # Calculate delta of volume with window 3
    delta_volume = df['volume'].diff(periods=3)
    
    # Rolling rank of delta volume with window 10
    rolling_rank = delta_volume.rolling(window=10, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Calculate correlation between absolute price change and volume change
    abs_delta_close = df['close'].diff().abs()
    delta_volume_1 = df['volume'].diff()
    
    # Rolling correlation with window 10
    correlation = abs_delta_close.rolling(window=10, min_periods=1).corr(delta_volume_1)
    
    # Combine factors
    factor = normalized_ratio + rolling_rank + correlation
    
    return factor
