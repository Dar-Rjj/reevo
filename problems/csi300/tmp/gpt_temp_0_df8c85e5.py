import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate intraday range
    df['intraday_range'] = df['high'] - df['low']
    
    # Normalize by historical volatility (5-day rolling median of range)
    df['median_range'] = df['intraday_range'].rolling(window=5, min_periods=1).median()
    df['normalized_range'] = df['intraday_range'] / df['median_range']
    
    # Calculate volume efficiency
    df['volume_efficiency'] = df['volume'] / df['intraday_range']
    
    # Normalize volume efficiency using z-score and clip
    df['volume_zscore'] = df['volume_efficiency'].rolling(window=5, min_periods=1).apply(lambda x: zscore(x)[-1], raw=True)
    df['volume_zscore'] = np.clip(df['volume_zscore'], -3, 3)
    
    # Calculate momentum
    df['momentum'] = df['close'].shift(1) - df['close'].shift(5)
    df['abs_momentum'] = abs(df['momentum'])
    
    # Combine signals
    df['factor'] = (df['normalized_range'] * df['volume_zscore']) / df['abs_momentum']
    
    # Return the factor series
    return df['factor']
