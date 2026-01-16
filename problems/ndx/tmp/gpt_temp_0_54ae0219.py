import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import skew

def heuristics_v2(df):
    # Calculate intraday range ratio as a percentage
    range_ratio = (df['high'] - df['low']) / df['close'] * 100
    
    # Calculate previous day range ratio
    prev_range_ratio = range_ratio.shift(1)
    
    # Compute range deviation
    range_deviation = range_ratio - prev_range_ratio
    
    # Calculate rolling skewness of volume over 10 days
    volume_skewness = df['volume'].rolling(window=10).apply(lambda x: skew(x), raw=True)
    
    # Apply log transformation to skewness values
    log_skewness = np.log1p(np.abs(volume_skewness)) * np.sign(volume_skewness)
    
    # Combine signals by multiplying range deviation by log-transformed skewness
    factor = range_deviation * log_skewness
    
    # Apply sign function to the final factor values
    factor = np.sign(factor)
    
    return factor
