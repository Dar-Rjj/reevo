import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate price range
    price_range = df['high'] - df['low']
    
    # Normalize by close and apply sign(close - open)
    normalized_momentum = (price_range / df['close']) * np.sign(df['close'] - df['open'])
    
    # Calculate volume change
    volume_ma = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_change = df['volume'] / volume_ma
    
    # Combine signals
    combined_signal = normalized_momentum * volume_change
    
    # Apply 3-day Z-score normalization
    z_score_signal = combined_signal.rolling(window=3, min_periods=1).apply(lambda x: zscore(x)[-1])
    
    return z_score_signal
