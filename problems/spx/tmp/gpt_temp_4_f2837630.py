import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Initialize result Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate components for each day
    for i in range(len(data)):
        if i < 5:  # Need at least 5 days for rolling mean
            factor.iloc[i] = np.nan
            continue
            
        current = data.iloc[i]
        past_volumes = data.iloc[i-5:i]['volume']
        
        # Price Stability Component
        normalized_range = (current['high'] - current['low']) / current['close']
        normalized_oc = abs(current['close'] - current['open']) / current['open']
        price_stability = (normalized_range + normalized_oc) / 2
        
        # Volume Confirmation
        vol_mean = past_volumes.mean()
        vol_deviation = current['volume'] - vol_mean
        
        # Combine components
        factor.iloc[i] = price_stability * vol_deviation
    
    return factor
