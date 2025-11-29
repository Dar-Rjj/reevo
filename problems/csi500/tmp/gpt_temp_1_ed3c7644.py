import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Opening Momentum Efficiency with Historical Memory
    """
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic price metrics
    data['prev_close'] = data.groupby(level=1)['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['abs_gap'] = np.abs(data['overnight_gap'])
    
    # Historical price anchors (5, 10, 20 days)
    for window in [5, 10, 20]:
        data[f'high_{window}d'] = data.groupby(level=1)['high'].rolling(window, min_periods=1).mean().reset_index(level=1, drop=True)
        data[f'low_{window}d'] = data.groupby(level=1)['low'].rolling(window, min_periods=1).mean().reset_index(level=1, drop=True)
        data[f'close_{window}d'] = data.groupby(level=1)['close'].rolling(window, min_periods=1).mean().reset_index(level=1, drop=True)
    
    # Gap behavior near historical levels
    gap_memory_interaction = 0
    for window in [5, 10, 20]:
        high_dist = (data['open'] - data[f'high_{window}d']) / data[f'high_{window}d']
        low_dist = (data['open'] - data[f'low_{window}d']) / data[f'low_{window}d']
        close_dist = (data['open'] - data[f'close_{window}d']) / data[f'close_{window}d']
        
        # Gap attraction/repulsion measure
        gap_attraction = np.where(data['overnight_gap'] > 0, 
                                 -np.abs(high_dist) + np.abs(low_dist),
                                 np.abs(high_dist) - np.abs(low_dist))
        
        gap_memory_interaction += gap_attraction * (1/window)  # Time decay
    
    # Opening range efficiency (simplified - using first hour as proxy)
    data['first_hour_high'] = data.groupby(level=1)['high'].rolling(2, min_periods=1).max().reset_index(level=1, drop=True)
    data['first_hour_low'] = data.groupby(level=1)['low'].rolling(2, min_periods=1).min().reset_index(level=1, drop=True)
    data['first_hour_close'] = data.groupby(level=1)['close'].shift(-1).fillna(data['close'])
    
    opening_range = data['first_hour_high'] - data['first_hour_low']
    opening_movement = np.abs(data['first_hour_close'] - data['open'])
    range_efficiency = np.where(opening_range > 0, opening_movement / opening_range, 0)
    
    # Range behavior near price anchors
    range_memory_confirmation = 0
    for window in [5, 10]:
        anchor_range = data[f'high_{window}d'] - data[f'low_{window}d']
        current_vs_anchor = opening_range / anchor_range
        range_memory_confirmation += np.where(current_vs_anchor < 1.5, range_efficiency, -range_efficiency) * (1/window)
    
    # Volume concentration analysis
    data['prev_volume_5d'] = data.groupby(level=1)['volume'].rolling(5, min_periods=1).mean().reset_index(level=1, drop=True)
    volume_concentration = np.where(data['prev_volume_5d'] > 0, 
                                  data['volume'] / data['prev_volume_5d'], 1)
    
    # Volume at extremes
    gap_volume_alignment = volume_concentration * np.abs(data['overnight_gap'])
    range_volume_alignment = volume_concentration * range_efficiency
    volume_opening_alignment = gap_volume_alignment + range_volume_alignment
    
    # Amount efficiency
    data['prev_amount_5d'] = data.groupby(level=1)['amount'].rolling(5, min_periods=1).mean().reset_index(level=1, drop=True)
    amount_concentration = np.where(data['prev_amount_5d'] > 0, 
                                  data['amount'] / data['prev_amount_5d'], 1)
    
    amount_efficiency = np.where(data['amount'] > 0, 
                               (data['first_hour_close'] - data['open']) / data['amount'], 0)
    
    amount_volume_sync = np.corrcoef(amount_concentration, volume_concentration)[0,1] if len(data) > 1 else 0
    amount_volume_opening_score = amount_efficiency * amount_volume_sync
    
    # Memory-momentum alignment
    memory_momentum_convergence = 0
    for window in [5, 10, 20]:
        momentum_vs_memory = (data['overnight_gap'] * 
                            (data['open'] - data[f'close_{window}d']) / data[f'close_{window}d'])
        memory_momentum_convergence += momentum_vs_memory * (1/window)
    
    # Transition quality
    gap_persistence = np.where(np.sign(data['overnight_gap']) == np.sign(data['first_hour_close'] - data['open']), 
                              np.abs(data['overnight_gap']), -np.abs(data['overnight_gap']))
    
    momentum_decay = np.abs(data['overnight_gap']) - np.abs((data['first_hour_close'] - data['open']) / data['open'])
    transition_quality = gap_persistence - momentum_decay
    
    # Memory-transition alignment
    memory_transition_alignment = transition_quality * memory_momentum_convergence
    
    # Final composite factor
    opening_efficiency_core = (gap_memory_interaction * range_memory_confirmation * 
                              volume_opening_alignment * amount_volume_opening_score)
    
    memory_enhanced_factor = (opening_efficiency_core * memory_momentum_convergence * 
                             memory_transition_alignment)
    
    # Handle NaN values
    memory_enhanced_factor = memory_enhanced_factor.replace([np.inf, -np.inf], 0).fillna(0)
    
    return memory_enhanced_factor
