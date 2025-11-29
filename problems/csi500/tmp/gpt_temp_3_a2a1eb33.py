import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(5)
    
    # Gap Magnitude Analysis
    data['gap_magnitude'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_direction'] = np.sign(data['gap_magnitude'])
    
    # Multi-Timeframe Efficiency
    data['daily_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['daily_range_efficiency'] = data['daily_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # 5-day Cumulative Efficiency
    data['cumulative_efficiency_5d'] = data['daily_range_efficiency'].rolling(window=5, min_periods=3).sum()
    
    # 10-day Efficiency
    data['cumulative_efficiency_10d'] = data['daily_range_efficiency'].rolling(window=10, min_periods=5).sum()
    
    # Efficiency Momentum
    data['efficiency_momentum'] = data['cumulative_efficiency_5d'] / data['cumulative_efficiency_10d']
    data['efficiency_momentum'] = data['efficiency_momentum'].replace([np.inf, -np.inf], np.nan)
    
    # Volume Momentum
    data['volume_momentum'] = (data['volume'] / data['prev_volume']) - 1
    data['volume_momentum'] = data['volume_momentum'].replace([np.inf, -np.inf], np.nan)
    
    # Breakout Components
    data['range_breakout_up'] = (data['close'] > data['prev_high']).astype(int)
    data['range_breakout_down'] = (data['close'] < data['prev_low']).astype(int)
    data['breakout_magnitude'] = abs(data['close'] - data['prev_close']) / data['prev_close']
    
    # Volume-Confirmed Breakout
    data['volume_confirmed_breakout'] = data['breakout_magnitude'] * data['volume_momentum']
    
    # Gap-Weighted Efficiency
    data['gap_weighted_efficiency'] = data['daily_range_efficiency'] * (1 + abs(data['gap_magnitude']) * data['gap_direction'])
    
    # Volume-Enhanced Breakout
    data['volume_enhanced_breakout'] = data['breakout_magnitude'] * data['volume_momentum']
    
    # Efficiency-Momentum Divergence
    data['efficiency_momentum_divergence'] = data['efficiency_momentum'] * data['volume_momentum']
    
    # Breakout-Efficiency Correlation
    data['breakout_efficiency_correlation'] = data['gap_weighted_efficiency'] * data['volume_enhanced_breakout']
    
    # Primary Signal - Efficiency-Breakout Divergence
    data['efficiency_breakout_divergence'] = data['efficiency_momentum_divergence'] * data['breakout_efficiency_correlation']
    
    # Volume-Weighted Adjustment
    data['primary_signal'] = data['efficiency_breakout_divergence'] * data['volume_momentum']
    
    # Gap-Efficiency Persistence
    data['gap_efficiency_sign'] = np.sign(data['gap_weighted_efficiency'])
    data['gap_efficiency_persistence'] = 0
    
    for i in range(1, len(data)):
        if data['gap_efficiency_sign'].iloc[i] == data['gap_efficiency_sign'].iloc[i-1]:
            data['gap_efficiency_persistence'].iloc[i] = data['gap_efficiency_persistence'].iloc[i-1] + 1
        else:
            data['gap_efficiency_persistence'].iloc[i] = 0
    
    # Breakout Momentum
    data['breakout_direction'] = np.where(data['range_breakout_up'] == 1, 1, 
                                        np.where(data['range_breakout_down'] == 1, -1, 0))
    data['breakout_momentum'] = 0
    data['consecutive_breakout_days'] = 0
    
    for i in range(1, len(data)):
        if data['breakout_direction'].iloc[i] == data['breakout_direction'].iloc[i-1] and data['breakout_direction'].iloc[i] != 0:
            data['consecutive_breakout_days'].iloc[i] = data['consecutive_breakout_days'].iloc[i-1] + 1
        else:
            data['consecutive_breakout_days'].iloc[i] = 1 if data['breakout_direction'].iloc[i] != 0 else 0
        
        data['breakout_momentum'].iloc[i] = data['consecutive_breakout_days'].iloc[i] * data['breakout_magnitude'].iloc[i]
    
    # Final Alpha
    data['final_alpha'] = data['efficiency_breakout_divergence'] * (1 + data['gap_efficiency_persistence'] + data['breakout_momentum'])
    
    # Clean up infinite values and return
    data['final_alpha'] = data['final_alpha'].replace([np.inf, -np.inf], np.nan)
    
    return data['final_alpha']
