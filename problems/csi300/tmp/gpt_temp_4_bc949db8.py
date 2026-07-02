import numpy as np
def heuristics_v2(df):
    # Calculate Price-to-Volume Ratio
    price_volume_ratio = df['close'] / df['volume']
    
    # Compute Trend Direction (5-day slope)
    # Using linear regression slope over rolling window
    def calculate_slope(series):
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    trend_direction = price_volume_ratio.rolling(5).apply(calculate_slope, raw=False)
    trend_signal = np.where(trend_direction > 0, 1, -1)
    
    # Calculate Volume Acceleration (5-day percentage change)
    volume_pct_change = df['volume'].pct_change(periods=1)
    volume_acceleration = volume_pct_change.rolling(5).mean()
    
    # Combine signals
    factor = trend_signal * volume_acceleration
    
    return factor
