import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Compute Price Reversal Component
    # Daily Price Change (Close - Open)
    daily_price_change = data['close'] - data['open']
    
    # Previous Day's Price Range (High - Low)
    prev_price_range = (data['high'] - data['low']).shift(1)
    
    # Price Reversal Component (avoid division by zero)
    price_reversal = daily_price_change / prev_price_range.replace(0, np.nan)
    
    # 2. Incorporate Volume Dynamics
    # Volume Acceleration (current volume / 5-day average volume)
    volume_5d_avg = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_acceleration = data['volume'] / volume_5d_avg.replace(0, np.nan)
    
    # Volume Trend Slope (linear regression slope of past 8 days volume)
    def volume_slope(series):
        if len(series) < 3:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = stats.linregress(x, series)
        return slope
    
    volume_trend_slope = data['volume'].rolling(window=8, min_periods=3).apply(
        volume_slope, raw=False
    )
    
    # 3. Measure Price Acceleration
    # Price Velocity (3-day rate of change of Close)
    price_velocity = data['close'].pct_change(periods=3)
    
    # Acceleration Change (current velocity - previous velocity)
    acceleration_change = price_velocity - price_velocity.shift(1)
    
    # 4. Combine Components with Interaction Terms
    # Multiply Reversal by Volume Acceleration
    reversal_volume_interaction = price_reversal * volume_acceleration
    
    # Multiply Result by Price Acceleration
    combined_signal = reversal_volume_interaction * acceleration_change
    
    # Apply Volume Trend Direction Filter
    # Take Sign of Volume Trend Slope
    volume_trend_direction = np.sign(volume_trend_slope)
    
    # Multiply by Combined Signal
    final_factor = combined_signal * volume_trend_direction
    
    return final_factor
