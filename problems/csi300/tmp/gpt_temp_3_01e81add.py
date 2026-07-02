import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Overnight Price Gap
    prev_close = df['close'].shift(1)
    gap_size = (df['open'] - prev_close) / prev_close
    normalized_gap = gap_size.ewm(span=5, adjust=False).mean()
    
    # Calculate Volume Anomaly
    volume_ma = df['volume'].rolling(window=20).mean()
    volume_anomaly = df['volume'] / volume_ma
    
    # Create Composite Signal
    composite_signal = normalized_gap * volume_anomaly
    
    # Calculate Recent Trend Direction
    def linear_slope(x):
        return np.polyfit(np.arange(len(x)), x, 1)[0]
    
    trend_direction = df['close'].rolling(window=5).apply(linear_slope)
    
    # Weight Final Signal
    final_signal = composite_signal * np.sign(trend_direction)
    
    return final_signal
