import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(data):
    # Make a copy to avoid SettingWithCopyWarning
    df = data.copy()
    
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Trend Strength Component
    # 10-day linear slope of close prices
    def linear_slope(series):
        x = np.arange(len(series))
        y = series.values
        if len(y) < 2:
            return np.nan
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    trend = df['close'].rolling(window=10, min_periods=10).apply(linear_slope, raw=False)
    # Normalize by close price at t-10
    normalized_trend = trend / df['close'].shift(10)
    
    # Volatility Adjustment
    vol = returns.rolling(window=10, min_periods=10).std()
    trend_strength = normalized_trend / (vol + 1e-6)  # Add small constant to avoid division by zero
    
    # Liquidity Confirmation
    # Calculate liquidity ratio (current volume / 20-day average volume)
    avg_volume = df['volume'].rolling(window=20, min_periods=20).mean()
    liquidity_ratio = df['volume'] / (avg_volume + 1e-6)
    
    # Combine components
    combined_factor = trend_strength * liquidity_ratio
    
    # Z-score normalization
    rolling_mean = combined_factor.rolling(window=20, min_periods=20).mean()
    rolling_std = combined_factor.rolling(window=20, min_periods=20).std()
    z_score = (combined_factor - rolling_mean) / (rolling_std + 1e-6)
    
    return z_score
