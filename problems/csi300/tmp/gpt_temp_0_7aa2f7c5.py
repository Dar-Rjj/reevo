import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(data):
    # Calculate intraday momentum (high-low range normalized by open)
    high_low_range = data['high'] - data['low']
    normalized_momentum = high_low_range / data['open']
    
    # Calculate volume ratio (current volume / 5-day average volume)
    # Using rolling window with min_periods=1 to handle initial days
    avg_volume_5day = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_ratio = data['volume'] / avg_volume_5day
    
    # Combine signals by multiplying momentum with volume ratio
    combined_signal = normalized_momentum * volume_ratio
    
    # Apply 3-day z-score normalization
    # Using rolling window with min_periods=1 to handle initial days
    zscore_signal = combined_signal.rolling(window=3, min_periods=1).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return zscore_signal
