import numpy as np
def heuristics_v2(df):
    # Price Direction Strength
    # Calculate High/Low Range
    high_low_range = df['high'] - df['low']
    normalized_range = high_low_range / df['close']
    
    # Calculate Close Position
    close_position = (df['close'] - df['low']) / (high_low_range + 1e-6)  # avoid division by zero
    close_position_bias = close_position - 0.5
    
    # Volume Trend Confirmation
    # Volume Trend Direction
    volume_slope = df['volume'].rolling(5).apply(lambda x: (x[-1] - x[0]) / len(x) if len(x) == 5 else 0)
    volume_trend_direction = np.sign(volume_slope)
    
    # Combine Signals
    price_strength = close_position_bias * normalized_range
    combined_signal = price_strength * volume_trend_direction
    
    return combined_signal
