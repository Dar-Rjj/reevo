import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute Intraday Rejection Signal
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Avoid division by zero
    price_range = high - low
    price_range = price_range.replace(0, np.nan)
    
    upper_rejection = (high - close) / price_range
    lower_rejection = (close - low) / price_range
    net_rejection = lower_rejection - upper_rejection
    
    # Incorporate Volume Trend
    def volume_slope(vol_series):
        if len(vol_series) < 5:
            return np.nan
        x = np.arange(len(vol_series))
        slope, _, _, _, _ = linregress(x, vol_series)
        return slope
    
    # Calculate 5-day volume slope using rolling window
    volume_trend = volume.rolling(window=5, min_periods=5).apply(
        volume_slope, raw=False
    )
    
    # Multiply Net Rejection by Volume Trend
    volume_weighted_rejection = net_rejection * volume_trend
    
    # Combine with Momentum
    # Calculate previous day's return (using shift(1) to get yesterday's close)
    prev_close = close.shift(1)
    daily_return = (close - prev_close) / prev_close
    
    # Multiply Volume-Weighted Rejection by 1-day return
    factor = volume_weighted_rejection * daily_return
    
    return factor
