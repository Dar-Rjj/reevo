import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for volatility
    data['returns'] = data['close'].pct_change()
    
    # 1. Intraday Reversal Signal
    # Calculate Intraday Range
    data['intraday_range'] = data['high'] - data['low']
    
    # Calculate Price Position within Range (avoid division by zero)
    data['price_position'] = (data['close'] - data['low']) / np.where(data['intraday_range'] == 0, 1, data['intraday_range'])
    
    # Compute Reversal Strength
    data['reversal_strength'] = data['intraday_range'] * (0.5 - data['price_position'])
    data['reversal_signal'] = np.sign(data['reversal_strength']) * np.sqrt(np.abs(data['reversal_strength']))
    
    # 2. Volatility-Adjusted Volume
    # Calculate Rolling Standard Deviation of Returns (10-day)
    data['volatility_10d'] = data['returns'].rolling(window=10, min_periods=5).std()
    
    # Compute Volume Adjustment Factor (avoid division by zero)
    data['volume_adjustment'] = np.log1p(data['volume'] / np.where(data['volatility_10d'] == 0, 1, data['volatility_10d']))
    
    # 3. Gap Persistence Component
    # Calculate Overnight Gap
    data['overnight_gap'] = (data['open'] / data['close'].shift(1)) - 1
    
    # Calculate previous day's range
    data['prev_day_range'] = data['high'].shift(1) - data['low'].shift(1)
    
    # Calculate Gap-to-Range Ratio (avoid division by zero)
    data['gap_to_range_ratio'] = np.abs(data['open'] - data['close'].shift(1)) / np.where(data['prev_day_range'] == 0, 1, data['prev_day_range'])
    
    # Compute Gap Fading Signal using inverse hyperbolic sine
    data['gap_fading'] = np.arcsinh(data['overnight_gap'] * data['gap_to_range_ratio'])
    
    # 4. Combine All Components
    # Multiply Reversal Signal by Volume Adjustment
    data['temp_factor'] = data['reversal_signal'] * data['volume_adjustment']
    
    # Multiply by Gap Fading Signal
    data['raw_factor'] = data['temp_factor'] * data['gap_fading']
    
    # Apply cross-sectional rank transformation
    def cross_sectional_rank(series):
        return series.rank(pct=True) - 0.5
    
    data['factor'] = data.groupby(data.index)['raw_factor'].transform(cross_sectional_rank)
    
    return data['factor']
