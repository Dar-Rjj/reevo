import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Price Divergence Component
    midpoint = (df['high'] + df['low']) / 2
    midpoint_deviation = (df['close'] - midpoint) / midpoint
    
    # Normalize by Volatility
    volatility = df['close'].rolling(window=5, min_periods=1).std()
    normalized_deviation = midpoint_deviation / volatility.replace(0, 1)  # Avoid division by zero
    
    # Volume Divergence Component
    rolling_volume_mean = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_spike = df['volume'] / rolling_volume_mean.replace(0, 1)  # Avoid division by zero
    log_volume_spike = np.log(volume_spike + 1)
    
    # Robust Scaling
    combined_signal = normalized_deviation * log_volume_spike
    
    # Normalize Factor
    rolling_median = combined_signal.rolling(window=20, min_periods=1).median()
    rolling_iqr = combined_signal.rolling(window=20, min_periods=1).apply(lambda x: x.quantile(0.75) - x.quantile(0.25))
    normalized_factor = (combined_signal - rolling_median) / rolling_iqr.replace(0, 1)  # Avoid division by zero
    
    return normalized_factor
