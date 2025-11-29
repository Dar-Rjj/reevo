import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price Extremes
    high = df['high']
    low = df['low']
    midpoint = (high + low) / 2
    
    # Compute Price Rejection Signals
    high_low_range = high - low
    # Avoid division by zero
    high_low_range = high_low_range.replace(0, np.nan)
    
    upper_rejection = (high - df['close']) / high_low_range
    lower_rejection = (df['close'] - low) / high_low_range
    net_rejection = lower_rejection - upper_rejection
    
    # Incorporate Volume Pattern
    volume = df['volume']
    
    # Calculate Volume Trend using rolling window (5 days)
    # Use linear regression slope over rolling window
    def volume_slope(vol_series):
        if len(vol_series) < 2:
            return np.nan
        x = np.arange(len(vol_series))
        return np.polyfit(x, vol_series, 1)[0]
    
    volume_trend = volume.rolling(window=5, min_periods=2).apply(
        volume_slope, raw=False
    )
    
    # Multiply Net Rejection by Volume Trend
    volume_weighted_rejection = net_rejection * volume_trend
    
    # Combine with Previous Day's Return
    # Calculate 1-day return using shift(1) to get previous day's close
    prev_close = df['close'].shift(1)
    one_day_return = (df['close'] - prev_close) / prev_close
    
    # Multiply by current day's Volume-Weighted Rejection
    factor = volume_weighted_rejection * one_day_return
    
    return factor
