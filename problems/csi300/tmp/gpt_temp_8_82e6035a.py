import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate intraday momentum (high-low range normalized by close)
    high_low_range = df['high'] - df['low']
    normalized_range = high_low_range / df['close']
    
    # Calculate 5-day rolling mean of normalized range (using only past data)
    momentum = normalized_range.rolling(5, min_periods=1).mean()
    
    # Calculate volume trend (current volume vs 5-day MA)
    volume_ma = df['volume'].rolling(5, min_periods=1).mean()
    volume_change = df['volume'] / volume_ma
    
    # Combine signals and apply z-score normalization
    factor = momentum * volume_change
    factor = factor.groupby(factor.index.date).apply(lambda x: zscore(x, ddof=1))
    
    return pd.Series(factor, index=df.index)
