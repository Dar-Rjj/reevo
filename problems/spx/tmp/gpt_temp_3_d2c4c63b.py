import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Actual Price Move
    actual_price_move = df['close'] - df['open']
    
    # Potential Price Range
    potential_price_range = df['high'] - df['low']
    
    # Short-term Volume Spike
    short_term_volume_spike = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Long-term Volume Trend
    long_term_volume_trend = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Raw Divergence
    raw_divergence = (actual_price_move / potential_price_range) * (short_term_volume_spike - long_term_volume_trend)
    
    # Normalize Signal using Z-score over 15 days
    normalized_signal = raw_divergence.rolling(window=15).apply(lambda x: zscore(x, ddof=1)[-1])
    
    return normalized_signal
