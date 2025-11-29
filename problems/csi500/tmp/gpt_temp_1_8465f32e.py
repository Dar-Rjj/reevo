import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Volume-Weighted Asymmetric Range Efficiency
    upward_range = np.where(data['close'] > data['open'], 
                           (data['close'] - data['open']) / (data['high'] - data['low']), 0)
    downward_range = np.where(data['close'] < data['open'], 
                             (data['open'] - data['close']) / (data['high'] - data['low']), 0)
    
    volume_trend = data['volume'] / data['volume'].shift(1)
    volume_trend = volume_trend.fillna(1)  # Handle first day
    
    asymmetric_range = (upward_range - downward_range) * volume_trend
    
    # Fractal Momentum with Rejection Patterns
    price_fractal = (data['high'] - data['low']) / data['close']
    avg_volume_20 = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_fractal = data['volume'] / avg_volume_20
    
    min_oc = np.minimum(data['open'], data['close'])
    max_oc = np.maximum(data['open'], data['close'])
    net_rejection = ((min_oc - data['low']) / (data['high'] - data['low']) - 
                    (data['high'] - max_oc) / (data['high'] - data['low']))
    
    fractal_momentum = price_fractal * volume_fractal * (1 + net_rejection)
    
    # Momentum Regime Persistence
    close_open_diff = data['close'] - data['open']
    direction = np.sign(close_open_diff)
    
    # Calculate streak length
    streak_length = pd.Series(index=data.index, dtype=float)
    current_streak = 0
    current_direction = 0
    
    for i in range(len(data)):
        if i == 0:
            streak_length.iloc[i] = 1
            current_direction = direction.iloc[i]
            current_streak = 1
        else:
            if direction.iloc[i] == current_direction and current_direction != 0:
                current_streak += 1
            else:
                current_streak = 1
                current_direction = direction.iloc[i]
            streak_length.iloc[i] = current_streak
    
    # Calculate average range efficiency during streak
    range_efficiency = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    range_efficiency = range_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Rolling average of range efficiency over streak window
    avg_range_efficiency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        streak_len = int(streak_length.iloc[i])
        start_idx = max(0, i - streak_len + 1)
        avg_range_efficiency.iloc[i] = range_efficiency.iloc[start_idx:i+1].mean()
    
    momentum_persistence = streak_length * avg_range_efficiency
    
    # Opening Gap Resolution
    opening_gap = data['open'] - data['close'].shift(1)
    opening_gap = opening_gap.fillna(0)
    
    resolution_efficiency = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        gap = opening_gap.iloc[i]
        if gap > 0:
            resolution_efficiency.iloc[i] = (data['high'].iloc[i] - data['open'].iloc[i]) / abs(gap) if abs(gap) > 0 else 0
        elif gap < 0:
            resolution_efficiency.iloc[i] = (data['open'].iloc[i] - data['low'].iloc[i]) / abs(gap) if abs(gap) > 0 else 0
        else:
            resolution_efficiency.iloc[i] = 0
    
    volume_momentum = np.log(data['volume'] / data['volume'].shift(1))
    volume_momentum = volume_momentum.fillna(0)
    
    gap_resolution = resolution_efficiency * (1 + volume_momentum)
    
    # Combine all factors with equal weighting
    factor = (asymmetric_range + fractal_momentum + momentum_persistence + gap_resolution) / 4
    
    return factor
