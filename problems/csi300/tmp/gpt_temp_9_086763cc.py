import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Intraday Price Movement
    high_low_range = df['high'] - df['low']
    normalized_range = high_low_range / df['close']
    rolling_normalized_range = normalized_range.rolling(5, min_periods=1).mean()
    
    # Intraday Momentum Persistence
    momentum = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    momentum_persistence = momentum.rolling(5, min_periods=1).mean()
    
    # Volume Confirmation
    volume_avg = df['volume'].rolling(20, min_periods=1).mean()
    volume_spike = df['volume'] / volume_avg.replace(0, np.nan)
    
    # Combine signals
    combined_signal = rolling_normalized_range * momentum_persistence * volume_spike
    
    # Apply z-score normalization using only historical data
    factor = combined_signal.expanding().apply(lambda x: zscore(x, ddof=1)[-1] if len(x) > 1 else np.nan)
    
    return factor
