import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday midpoint
    data['midpoint'] = (data['high'] + data['low']) / 2
    
    # Determine daily trend direction (1 if close > midpoint, -1 if close < midpoint, 0 if equal)
    data['trend_direction'] = np.where(data['close'] > data['midpoint'], 1, 
                                     np.where(data['close'] < data['midpoint'], -1, 0))
    
    # Calculate trend direction persistence
    data['trend_persistence'] = 0
    for i in range(1, len(data)):
        if data['trend_direction'].iloc[i] == data['trend_direction'].iloc[i-1]:
            data['trend_persistence'].iloc[i] = data['trend_persistence'].iloc[i-1] + 1
        else:
            data['trend_persistence'].iloc[i] = 0
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate absolute close-to-open return
    data['abs_return'] = abs(data['close'] - data['open'])
    
    # Calculate price range efficiency
    data['price_efficiency'] = np.where(data['true_range'] > 0, 
                                      data['abs_return'] / data['true_range'], 0)
    
    # Calculate volume intensity (current volume vs 5-day average)
    data['avg_volume_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['avg_volume_5d']
    
    # Combine efficiency with volume confirmation
    data['momentum_efficiency'] = data['price_efficiency'] * data['volume_ratio']
    
    # Volume direction consistency
    data['volume_change'] = data['volume'] - data['volume'].shift(1)
    data['volume_direction'] = np.where(data['volume_change'] > 0, 1, 
                                      np.where(data['volume_change'] < 0, -1, 0))
    
    data['volume_persistence'] = 0
    for i in range(1, len(data)):
        if data['volume_direction'].iloc[i] == data['volume_direction'].iloc[i-1]:
            data['volume_persistence'].iloc[i] = data['volume_persistence'].iloc[i-1] + 1
        else:
            data['volume_persistence'].iloc[i] = 0
    
    # Volume-based persistence weight
    data['volume_weight'] = 1 + (data['volume_persistence'] * 0.1)
    
    # Time-decay adjustment
    data['high_volume_signal'] = (data['volume_ratio'] > 1.2) & (data['momentum_efficiency'] > 0.5)
    
    # Calculate days since last high-volume signal
    data['days_since_high_volume'] = 0
    last_high_volume_idx = -1
    for i in range(len(data)):
        if data['high_volume_signal'].iloc[i]:
            last_high_volume_idx = i
            data['days_since_high_volume'].iloc[i] = 0
        elif last_high_volume_idx >= 0:
            data['days_since_high_volume'].iloc[i] = i - last_high_volume_idx
        else:
            data['days_since_high_volume'].iloc[i] = 999  # Large number if no signal yet
    
    # Exponential decay factor
    data['time_decay'] = np.exp(-data['days_since_high_volume'] * 0.1)
    
    # Synthesize final alpha factor
    data['trend_momentum_raw'] = data['trend_persistence'] * data['momentum_efficiency']
    data['weighted_component'] = data['trend_momentum_raw'] * data['volume_weight']
    data['alpha_factor'] = data['weighted_component'] * data['time_decay']
    
    # Return the alpha factor series
    return data['alpha_factor']
