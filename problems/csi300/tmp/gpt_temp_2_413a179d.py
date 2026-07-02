import pandas as pd
import numpy as np
def heuristics_v2(data):
    # Calculate price deviation from midpoint
    midpoint = (data['high'] + data['low']) / 2
    price_deviation = abs(data['close'] - midpoint)
    
    # Normalize by daily range
    daily_range = data['high'] - data['low']
    normalized_deviation = price_deviation / (daily_range + 1e-6)  # Add small constant to avoid division by zero
    
    # Calculate efficiency score (1 - normalized deviation)
    efficiency_score = 1 - normalized_deviation
    
    # Calculate volume trend (5-day slope)
    volume = data['volume']
    rolling_window = 5
    # Create time index for regression
    time_index = np.arange(rolling_window)
    # Initialize slope array
    volume_slope = np.zeros(len(volume))
    # Calculate slope for each window
    for i in range(rolling_window - 1, len(volume)):
        window_volume = volume.iloc[i - rolling_window + 1:i + 1].values
        if len(window_volume) == rolling_window:
            slope = np.polyfit(time_index, window_volume, 1)[0]
            volume_slope[i] = slope
    
    # Calculate volume standard deviation over same window
    volume_std = volume.rolling(window=rolling_window, min_periods=1).std()
    
    # Calculate volume trend strength (slope normalized by std)
    volume_trend_strength = volume_slope / (volume_std + 1e-6)
    
    # Combine signals
    factor = efficiency_score * volume_trend_strength
    
    return pd.Series(factor, index=data.index)
