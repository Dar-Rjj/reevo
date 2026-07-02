import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate Gap
    gap = df['open'] - df['close'].shift(1)
    
    # Normalize Gap
    gap_normalized = gap / df['close'].shift(1)
    gap_zscore = gap_normalized.rolling(window=20).apply(lambda x: zscore(x)[-1], raw=False)
    
    # Measure Volume Surge
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / volume_ma
    
    # Combine Signals
    combined_signal = gap_zscore * volume_ratio
    
    # Apply Exponential Smoothing
    factor = combined_signal.ewm(alpha=0.1, adjust=False).mean()
    
    return factor.dropna()
