import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate midpoint and midpoint returns
    midpoint = (data['high'] + data['low']) / 2
    prev_close = data['close'].shift(1)
    midpoint_return = (midpoint - prev_close) / prev_close
    
    # Calculate intraday range (volatility)
    intraday_range = (data['high'] - data['low']) / prev_close
    intraday_range = intraday_range.replace(0, np.nan)  # Avoid division by zero
    
    # Compute volatility-adjusted momentum
    vol_adj_momentum = midpoint_return / intraday_range
    
    # Calculate volume trend
    volume_trend = pd.Series(index=data.index, dtype=float)
    window = 5
    for i in range(len(data)):
        if i >= window - 1:
            window_data = data['volume'].iloc[i-window+1:i+1]
            if not window_data.isna().any():
                slope = linregress(np.arange(window), window_data.values).slope
                volume_trend.iloc[i] = slope
    
    # Apply exponential smoothing to volume trend
    alpha = 0.3  # Smoothing factor
    smoothed_weight = volume_trend.ewm(alpha=alpha, adjust=False).mean()
    
    # Normalize smoothed weight to [0,1] range
    min_weight = smoothed_weight.rolling(window=252, min_periods=1).min()  # 1 year lookback
    max_weight = smoothed_weight.rolling(window=252, min_periods=1).max()
    normalized_weight = (smoothed_weight - min_weight) / (max_weight - min_weight + 1e-6)
    
    # Combine signals
    factor = vol_adj_momentum * normalized_weight
    
    return factor
