import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Measure Intraday Momentum
    high_low_range = df['high'] - df['low']
    normalized_momentum = high_low_range / df['open']
    
    # Confirm with Volume Trend
    current_volume = df['volume']
    moving_avg_volume = current_volume.rolling(window=5, min_periods=1).mean()
    volume_change = current_volume / moving_avg_volume
    
    # Combine Signals
    combined_signal = normalized_momentum * volume_change
    
    # Apply Z-score Normalization
    factor_values = combined_signal.groupby(level=0).apply(zscore)
    
    return factor_values
