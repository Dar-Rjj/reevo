import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Calculate gap size: open(t) - close(t-1)
    gap_size = data['open'] - data['close'].shift(1)
    
    # Normalize gap by previous close
    normalized_gap = gap_size / data['close'].shift(1)
    abs_normalized_gap = normalized_gap.abs()
    
    # Calculate 5-day volume trend strength
    volume_trend_strength = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data['volume'].iloc[i-4:i+1]
        if window.min() == window.max():  # handle zero slope case
            slope = 0
        else:
            slope = linregress(np.arange(5), window.values).slope
        mean_volume = window.mean()
        volume_trend_strength.iloc[i] = slope / mean_volume if mean_volume != 0 else 0
    
    # Combine signals
    factor = abs_normalized_gap * volume_trend_strength
    factor = factor * np.sign(normalized_gap)  # preserve original gap direction
    
    return factor
