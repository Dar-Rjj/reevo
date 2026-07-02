import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(data):
    # Calculate intraday range (high - low)
    intraday_range = data['high'] - data['low']
    
    # Normalize intraday range by open price
    normalized_range = intraday_range / data['open']
    
    # Calculate 5-day moving average of volume (using only past data)
    ma_volume = data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Calculate volume ratio (current volume / 5-day MA volume)
    volume_ratio = data['volume'] / ma_volume
    
    # Combine signals by multiplying normalized range and volume ratio
    combined_signal = normalized_range * volume_ratio
    
    # Apply z-score normalization using expanding window (only past data)
    factor = combined_signal.expanding().apply(lambda x: zscore(x)[-1] if len(x) > 1 else np.nan)
    
    return factor
