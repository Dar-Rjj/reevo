import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate momentum (close_t - close_t-5)
    momentum = data['close'].diff(5)
    
    # Calculate 5-day median volume
    median_volume = data['volume'].rolling(5, min_periods=1).median()
    
    for i in range(1, len(data)):
        current = data.iloc[i]
        previous = data.iloc[i-1]
        
        # Initialize reversal signal
        reversal_signal = 0
        
        # Price reversal detection
        if current['open'] > previous['close'] and momentum.iloc[i] < 0:
            reversal_signal = -1  # Bearish reversal
        elif current['open'] < previous['close'] and momentum.iloc[i] > 0:
            reversal_signal = 1   # Bullish reversal
        
        # Volume confirmation and weighting
        if median_volume.iloc[i] > 0:
            volume_ratio = min(current['volume'] / median_volume.iloc[i], 2.0)
            factor.iloc[i] = reversal_signal * volume_ratio
        else:
            factor.iloc[i] = 0
    
    return factor
