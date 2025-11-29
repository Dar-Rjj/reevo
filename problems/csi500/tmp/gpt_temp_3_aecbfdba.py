import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous day close
    data['prev_close'] = data['close'].shift(1)
    
    # Calculate first hour data (assuming first hour is first 1/6.5 of trading day)
    # For simplicity, we'll use the first available data point as proxy for first hour
    # In practice, you would use actual intraday data
    data['first_hour_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['first_hour_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['first_hour_close'] = data['close'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['first_hour_volume'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['first_hour_open'] = data['open'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    
    # Calculate previous day volume
    data['prev_volume'] = data['volume'].shift(1)
    
    # Opening Session Momentum - Gap Pressure
    gap = data['open'] - data['prev_close']
    volume_signal = np.sign(data['first_hour_volume'] - data['prev_volume'])
    gap_pressure = gap * volume_signal
    
    # Opening Session Momentum - Range Efficiency
    first_hour_range = data['first_hour_high'] - data['first_hour_low']
    open_close_gap = np.abs(data['open'] - data['prev_close'])
    # Avoid division by zero
    range_efficiency = np.where(open_close_gap > 0, first_hour_range / open_close_gap, 0)
    
    # Intraday Range Expansion - Breakout Quality
    close_first_hour_diff = data['close'] - data['first_hour_close']
    high_breakout = (data['high'] != data['first_hour_high']).astype(int)
    low_breakout = (data['low'] != data['first_hour_low']).astype(int)
    breakout_occurred = (high_breakout | low_breakout).astype(int)
    breakout_quality = close_first_hour_diff * breakout_occurred
    
    # Volume-Impact Dynamics - Opening Impact
    first_hour_price_change = np.abs(data['first_hour_close'] - data['first_hour_open'])
    # Avoid division by zero and handle very small volumes
    opening_impact = np.where(data['first_hour_volume'] > 0, 
                             first_hour_price_change / data['first_hour_volume'], 0)
    
    # Combine factors (simple average for demonstration)
    # In practice, you might want to weight these differently
    factor = (gap_pressure + range_efficiency + breakout_quality + opening_impact) / 4
    
    return pd.Series(factor, index=data.index)
