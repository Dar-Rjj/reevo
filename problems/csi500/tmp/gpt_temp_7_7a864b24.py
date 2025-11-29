import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    high_low = data['high'] - data['low']
    high_prev_close = abs(data['high'] - data['close'].shift(1))
    low_prev_close = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Calculate Price Movement Efficiency
    abs_close_open_change = abs(data['close'] - data['open'])
    price_movement_efficiency = abs_close_open_change / true_range
    price_movement_efficiency = price_movement_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Calculate Volume Acceleration
    rolling_median_volume = data['volume'].rolling(window=20, min_periods=10).median()
    volume_ratio = data['volume'] / rolling_median_volume
    volume_acceleration = volume_ratio - volume_ratio.shift(1)
    
    # Calculate Volume Trend Direction
    price_direction = np.where(data['close'] > data['open'], 1, 
                              np.where(data['close'] < data['open'], -1, 0))
    volume_confirmation = volume_acceleration * price_direction
    
    # Combine Trend and Volume Signals
    combined_signal = price_movement_efficiency * volume_confirmation
    
    # Apply Rolling Mean Adjustment
    rolling_mean = combined_signal.rolling(window=5, min_periods=3).mean()
    rolling_mad = combined_signal.rolling(window=5, min_periods=3).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))) if len(x) >= 3 else np.nan
    )
    
    normalized_signal = (combined_signal - rolling_mean) / rolling_mad
    normalized_signal = normalized_signal.replace([np.inf, -np.inf], np.nan)
    
    # Apply Price Momentum Filter
    short_term_return = data['close'].pct_change(periods=3)
    momentum_direction = np.sign(short_term_return)
    momentum_filtered = normalized_signal * momentum_direction
    
    # Apply Volatility Normalization
    rolling_volatility = momentum_filtered.rolling(window=10, min_periods=5).std()
    final_factor = momentum_filtered / rolling_volatility
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    
    return final_factor
