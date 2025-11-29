import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Calculate intraday metrics
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['intraday_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Intraday Price Acceleration components
    # Current intraday return slope (using rolling regression)
    def rolling_slope(x):
        if len(x) < 3:
            return np.nan
        try:
            idx = np.arange(len(x))
            return np.polyfit(idx, x, 1)[0]
        except:
            return np.nan
    
    # Calculate rolling intraday return slope (3-day window)
    data['intraday_return_3d'] = data['intraday_return'].rolling(window=3, min_periods=2).apply(rolling_slope, raw=False)
    
    # Recent acceleration trend (change in slope)
    data['acceleration_trend'] = data['intraday_return_3d'].diff(2)
    
    # Volume Confirmation Pattern
    # Volume trend alignment (5-day volume slope)
    data['volume_slope_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(rolling_slope, raw=False)
    
    # Volume distribution characteristics
    data['volume_ma_ratio'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_std_ratio'] = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / data['volume'].rolling(window=20, min_periods=10).std()
    
    # Signal Integration
    # Acceleration-Volume congruence scoring
    data['accel_volume_congruence'] = (
        np.sign(data['intraday_return_3d']) * np.sign(data['volume_slope_5d']) * 
        np.abs(data['intraday_return_3d']) * np.abs(data['volume_slope_5d'])
    )
    
    # Multi-timeframe confirmation
    # Short-term (3-day) vs medium-term (5-day) momentum alignment
    data['short_term_momentum'] = data['close'].pct_change(3)
    data['medium_term_momentum'] = data['close'].pct_change(5)
    data['momentum_alignment'] = np.sign(data['short_term_momentum']) * np.sign(data['medium_term_momentum'])
    
    # Combine signals with weights
    data['factor'] = (
        0.4 * data['accel_volume_congruence'].fillna(0) +
        0.3 * data['acceleration_trend'].fillna(0) +
        0.2 * data['volume_ma_ratio'].fillna(0) +
        0.1 * data['momentum_alignment'].fillna(0)
    )
    
    # Final factor series
    factor_series = data['factor']
    
    return factor_series
