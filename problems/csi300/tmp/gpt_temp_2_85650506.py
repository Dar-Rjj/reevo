import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Reversal Component
    # Calculate Prior Day Return
    prior_return = df['close'].diff()
    
    # Normalize by Average Range
    rolling_range = df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()
    normalized_return = prior_return / rolling_range
    
    # Volume Scaling Component
    # Calculate Volume Percentile
    volume_percentile = df['volume'].rolling(window=50).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min()))
    
    # Apply Square Root Transformation
    volume_scaling = np.sqrt(volume_percentile)
    
    # Combine Components
    factor = normalized_return * volume_scaling
    
    return factor.dropna()
