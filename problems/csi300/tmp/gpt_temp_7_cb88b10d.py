import pandas as pd
import pandas as pd
from scipy.stats import percentileofscore

def heuristics_v2(data):
    # Calculate Intraday Momentum
    intraday_momentum = (data['high'] - data['low']) / data['open']
    
    # Calculate Volume Percentile
    volume_percentile = data['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: percentileofscore(x, x.iloc[-1]) / 100, raw=False
    )
    
    # Scale Momentum by Percentile
    scaled_momentum = intraday_momentum * volume_percentile
    
    # Normalize to [-1, 1]
    normalized_momentum = 2 * (scaled_momentum - scaled_momentum.min()) / (scaled_momentum.max() - scaled_momentum.min()) - 1
    
    return normalized_momentum
