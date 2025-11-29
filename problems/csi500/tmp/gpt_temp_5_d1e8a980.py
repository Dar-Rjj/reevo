import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Gap Size vs Range Context
    data['gap_size'] = data['open'] - data['prev_close']
    data['prev_range'] = data['prev_high'] - data['prev_low']
    data['gap_to_range_ratio'] = data['gap_size'] / data['prev_range'].replace(0, np.nan)
    
    # Intraday Gap Persistence (simplified - using hourly data approximation)
    # For morning gap hold: first hour high vs open
    data['morning_gap_hold'] = np.sign(data['high'].rolling(window=2, min_periods=2).apply(lambda x: x.iloc[1] - data.loc[x.index[1], 'open'], raw=False)) * np.sign(data['gap_size'])
    # For afternoon gap confirmation: close vs afternoon low (using last 2 hours approximation)
    data['afternoon_gap_confirmation'] = np.sign(data['close'] - data['low'].rolling(window=2, min_periods=2).apply(lambda x: x.min(), raw=False)) * np.sign(data['gap_size'])
    data['gap_persistence_score'] = data['morning_gap_hold'].fillna(0) + data['afternoon_gap_confirmation'].fillna(0)
    
    # Range Expansion-Compression Around Gaps
    data['current_range'] = data['high'] - data['low']
    data['range_change'] = data['current_range'] / data['prev_range'].replace(0, np.nan)
    data['gap_driven_expansion'] = np.sign(data['gap_size']) * np.sign(data['range_change'] - 1)
    data['compression_breakout_potential'] = (1 - data['range_change']) * abs(data['gap_size'])
    
    # Volume-Weighted Gap Momentum
    data['gap_volume_intensity'] = data['volume'] * abs(data['gap_size'])
    data['volume_surge'] = data['volume'] - data['prev_volume']
    data['volume_gap_alignment'] = np.sign(data['gap_size']) * np.sign(data['volume_surge'])
    
    # Intraday Momentum Consistency (simplified)
    data['morning_range_efficiency'] = (data['high'].rolling(window=2, min_periods=2).apply(lambda x: x.iloc[1] - data.loc[x.index[1], 'low'], raw=False)) / data['current_range'].replace(0, np.nan)
    data['afternoon_persistence'] = (data['high'] - data['low'].rolling(window=2, min_periods=2).apply(lambda x: x.min(), raw=False)) / data['current_range'].replace(0, np.nan)
    data['session_alignment'] = np.sign(data['morning_range_efficiency'] - 0.5) * np.sign(data['afternoon_persistence'] - 0.5)
    data['momentum_quality'] = data['session_alignment'] * data['volume_gap_alignment']
    
    # Gap Filling vs Extension Behavior
    data['gap_fill_progress'] = (data['close'] - data['open']) / data['gap_size'].replace(0, np.nan)
    data['partial_fill_efficiency'] = abs(data['gap_fill_progress']) * (1 - abs(data['gap_fill_progress']))
    data['extension_momentum'] = np.sign(data['gap_fill_progress']) * np.sign(data['gap_size'])
    
    # Range Breakout Confirmation
    data['range_utilization'] = data['current_range'] / data['prev_range'].replace(0, np.nan)
    data['breakout_direction'] = np.sign(data['high'] - data['prev_high']) + np.sign(data['low'] - data['prev_low'])
    data['volume_breakout_confirmation'] = np.sign(data['breakout_direction']) * np.sign(data['volume_surge'])
    data['range_breakout_quality'] = data['breakout_direction'] * data['volume_breakout_confirmation']
    
    # Adaptive Factor Components
    data['gap_strength'] = data['gap_size'] * data['gap_persistence_score']
    data['volume_confirmation'] = data['volume_gap_alignment'] * data['volume_surge']
    data['raw_signal'] = data['gap_strength'] * data['volume_confirmation']
    
    data['range_change_direction'] = np.sign(data['range_change'] - 1)
    data['gap_range_synergy'] = data['gap_driven_expansion'] * data['range_breakout_quality']
    data['efficiency_score'] = data['range_change_direction'] * data['gap_range_synergy']
    
    # Generate Final Alpha Factor
    data['core_momentum_component'] = data['raw_signal'] * data['momentum_quality']
    data['regime_adjustment'] = data['partial_fill_efficiency'] * data['extension_momentum']
    data['final_factor'] = data['core_momentum_component'] * data['efficiency_score'] * data['regime_adjustment']
    
    # Return the final factor series
    return data['final_factor']
