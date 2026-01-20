import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(data):
    # Price-Range Divergence components
    # Current Day Range Ratio: (High - Low) / 5-day SMA(High - Low)
    daily_range = data['high'] - data['low']
    sma5_range = daily_range.rolling(window=5, min_periods=5).mean()
    range_ratio = daily_range / sma5_range
    
    # Price Deviation: (Close / 5-day SMA(Close)) - 1
    sma5_close = data['close'].rolling(window=5, min_periods=5).mean()
    price_deviation = (data['close'] / sma5_close) - 1
    
    # Combine to get Price-Range Divergence
    price_range_divergence = range_ratio * price_deviation
    
    # Volume Acceleration components
    # Volume Trend: Volume / 10-day SMA(Volume)
    sma10_volume = data['volume'].rolling(window=10, min_periods=10).mean()
    volume_trend = data['volume'] / sma10_volume
    
    # Volume Momentum: 3-day Slope of Volume divided by Volume Trend
    # Calculate slope using linear regression over last 3 days
    def calculate_slope(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        return slope
    
    volume_slope = data['volume'].rolling(window=3).apply(calculate_slope, raw=True)
    volume_momentum = volume_slope / volume_trend
    
    # Combine to get Volume Acceleration
    volume_acceleration = volume_trend * volume_momentum
    
    # Combined Signal: Multiply components and apply 5-day rolling z-score
    combined_signal = price_range_divergence * volume_acceleration
    z_score = combined_signal.rolling(window=5, min_periods=5).apply(
        lambda x: zscore(x, ddof=1)[-1] if len(x) == 5 else np.nan
    )
    
    return z_score
