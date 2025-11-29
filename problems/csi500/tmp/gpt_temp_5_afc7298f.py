import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Dynamics Assessment
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_volatility'] = data['overnight_gap'].rolling(window=20).std()
    data['gap_extreme'] = np.abs(data['overnight_gap']) > (2 * data['gap_volatility'])
    
    # Volatility Compression Detection
    data['prev_close_for_tr'] = data['close'].shift(1)
    data['tr_hl'] = data['high'] - data['low']
    data['tr_hc'] = np.abs(data['high'] - data['prev_close_for_tr'])
    data['tr_lc'] = np.abs(data['low'] - data['prev_close_for_tr'])
    data['true_range'] = np.maximum(data['tr_hl'], np.maximum(data['tr_hc'], data['tr_lc']))
    data['tr_10d_avg'] = data['true_range'].rolling(window=10).mean()
    data['compression_ratio'] = data['true_range'] / data['tr_10d_avg']
    
    # Compression persistence
    data['tr_decreasing'] = data['true_range'] < data['true_range'].shift(1)
    data['compression_persistence'] = data['tr_decreasing'].rolling(window=5, min_periods=1).apply(
        lambda x: x[::-1].cumprod()[::-1].sum(), raw=False
    )
    
    # Momentum Convergence Framework
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_convergence'] = (data['intraday_momentum'] * data['medium_term_momentum']) > 0
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['momentum_divergence'] = (data['momentum_3d'] - data['momentum_10d']) * data['momentum_convergence'].astype(int)
    
    # Volume Dynamics Integration
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_concentration'] = data['volume'] / data['volume_5d_avg']
    
    # Volume acceleration (3-day slope)
    data['volume_3d_avg'] = data['volume'].rolling(window=3).mean()
    data['volume_acceleration'] = (data['volume'] - data['volume_3d_avg']) / data['volume_3d_avg']
    
    data['volume_momentum_alignment'] = data['volume_concentration'] * data['momentum_divergence']
    
    # Range Utilization Analysis
    data['intraday_range'] = data['high'] - data['low']
    data['range_utilization'] = (data['close'] - data['low']) / data['intraday_range']
    data['gap_closure_progress'] = (data['close'] - data['open']) / (data['open'] - data['prev_close'])
    data['true_range_efficiency'] = np.abs(data['close'] - data['open']) / data['true_range']
    
    # Signal Construction Engine
    # Core Convergence Signal
    gap_direction = np.sign(data['overnight_gap'])
    data['gap_volatility_interaction'] = -gap_direction * data['compression_ratio']
    data['momentum_volume_alignment'] = data['volume_momentum_alignment'] * data['range_utilization']
    data['convergence_multiplier'] = 1 + data['momentum_convergence'].astype(int)
    
    data['core_convergence'] = (data['gap_volatility_interaction'] + data['momentum_volume_alignment']) * data['convergence_multiplier']
    
    # Efficiency Enhancement
    data['range_efficiency_scaling'] = data['true_range_efficiency'] * data['gap_closure_progress']
    data['compression_persistence_weighting'] = data['compression_persistence'] * data['core_convergence']
    data['volume_acceleration_confirmation'] = data['volume_acceleration'] * data['momentum_divergence']
    
    # Multi-Dimensional Integration
    data['combined_convergence'] = data['core_convergence'] * data['convergence_multiplier']
    data['efficiency_weighted'] = data['combined_convergence'] * data['range_efficiency_scaling']
    data['volume_aligned_factor'] = data['efficiency_weighted'] * data['volume_acceleration_confirmation']
    
    # Conditional Factor Generation
    # Signal Application Rules
    enhanced_convergence_condition = data['gap_extreme'] & (data['compression_ratio'] < 0.8)
    momentum_volume_priority_condition = data['volume_concentration'] > 1.2
    range_efficiency_emphasis_condition = data['true_range_efficiency'] > 0.6
    
    # Dynamic Weighting System
    compression_scaling = 1 + (data['compression_persistence'] * 0.1)
    volume_acceleration_adjustment = 1 + (data['volume_acceleration'] * 0.05)
    gap_absorption_weighting = 1 + (np.abs(data['gap_closure_progress']) * 0.2)
    
    # Final Factor Construction
    base_factor = data['volume_aligned_factor'].copy()
    
    # Apply conditional enhancements
    base_factor[enhanced_convergence_condition] = base_factor[enhanced_convergence_condition] * 1.5
    base_factor[momentum_volume_priority_condition] = base_factor[momentum_volume_priority_condition] * 1.3
    base_factor[range_efficiency_emphasis_condition] = base_factor[range_efficiency_emphasis_condition] * 1.2
    
    # Apply dynamic weighting
    final_factor = base_factor * compression_scaling * volume_acceleration_adjustment * gap_absorption_weighting
    
    # Clean infinite values and handle NaN
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    
    return final_factor
