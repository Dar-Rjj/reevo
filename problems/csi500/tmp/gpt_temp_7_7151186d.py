import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Price and Range Metrics
    df = df.copy()
    df['daily_range'] = df['high'] - df['low']
    df['midpoint'] = (df['high'] + df['low']) / 2
    df['deviation_from_midpoint'] = df['close'] - df['midpoint']
    
    # Calculate rolling midpoint extremes
    df['midpoint_5d_max'] = df['midpoint'].rolling(window=5, min_periods=1).max()
    df['midpoint_5d_min'] = df['midpoint'].rolling(window=5, min_periods=1).min()
    
    # Assess Range Expansion Consistency
    df['range_expanding'] = df['daily_range'] > df['daily_range'].shift(1)
    df['range_streak'] = 0
    current_streak = 0
    
    for i in range(len(df)):
        if i == 0:
            df.iloc[i, df.columns.get_loc('range_streak')] = 0
            continue
            
        if df['range_expanding'].iloc[i]:
            current_streak += 1
        else:
            current_streak = 0
            
        df.iloc[i, df.columns.get_loc('range_streak')] = current_streak
    
    df['range_expansion_consistency'] = 0.9 ** df['range_streak']
    
    # Analyze Volume-Weighted Confirmation Patterns
    df['high_volume_weighted'] = df['high'] * df['volume']
    df['low_volume_weighted'] = df['low'] * df['volume']
    df['extreme_ratio'] = df['high_volume_weighted'] / (df['low_volume_weighted'] + 1e-8)
    
    # Calculate Volume Acceleration
    df['volume_5d_avg'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_acceleration'] = df['volume'] / (df['volume_5d_avg'] + 1e-8)
    
    # Assess Volume Trend
    df['volume_trend'] = np.sign(df['volume'] - df['volume'].shift(1))
    df['volume_trend'].fillna(0, inplace=True)
    
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['true_range'].fillna(df['high'] - df['low'], inplace=True)
    
    # Synthesize Composite Factor
    # Breakout Logic
    denominator = df['midpoint_5d_max'] - df['midpoint_5d_min']
    denominator = np.where(denominator == 0, 1, denominator)
    breakout_signal = (df['midpoint'] - df['midpoint_5d_min']) / denominator
    
    # Incorporate Range Persistence
    factor = breakout_signal * df['range_expansion_consistency']
    
    # Apply Volume-Weighted Confirmation
    factor = factor * df['volume_acceleration']
    
    # Add Volume Trend Adjustment
    factor = factor * df['volume_trend']
    
    # Scale by True Range Component
    factor = factor * df['true_range']
    
    return pd.Series(factor, index=df.index)
