import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Session Gap-Range Momentum with Volume Efficiency Divergence alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['gap_magnitude'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Estimate first hour close as average of first hour high/low (proxy)
    data['first_hour_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x.max() if len(x) == 2 else np.nan)
    data['first_hour_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x.min() if len(x) == 2 else np.nan)
    data['first_hour_close'] = (data['first_hour_high'] + data['first_hour_low']) / 2
    
    # Morning session components
    data['morning_range'] = data['first_hour_high'] - data['first_hour_low']
    data['morning_range_position'] = np.where(
        data['morning_range'] > 0,
        (data['first_hour_close'] - data['first_hour_low']) / data['morning_range'],
        0.5
    )
    data['morning_gap_range_efficiency'] = data['gap_magnitude'] * data['morning_range_position']
    
    # Afternoon session components
    data['daily_range'] = data['high'] - data['low']
    data['afternoon_range_position'] = np.where(
        data['daily_range'] > 0,
        (data['close'] - data['low']) / data['daily_range'],
        0.5
    )
    
    # Previous session values for acceleration calculations
    data['prev_afternoon_range'] = data['afternoon_range_position'].shift(1)
    data['prev_morning_range'] = data['morning_range_position'].shift(1)
    
    # Range acceleration components
    data['afternoon_range_acceleration'] = np.where(
        data['prev_afternoon_range'] > 0,
        ((data['afternoon_range_position'] - data['prev_afternoon_range']) / data['prev_afternoon_range']) - 
        (data['prev_afternoon_range'] - data['prev_afternoon_range'].shift(1)),
        0
    )
    
    data['morning_range_acceleration'] = np.where(
        data['prev_morning_range'] > 0,
        ((data['morning_range_position'] - data['prev_morning_range']) / data['prev_morning_range']) - 
        (data['prev_morning_range'] - data['prev_morning_range'].shift(1)),
        0
    )
    
    # Gap persistence
    data['gap_persistence_ratio'] = np.where(
        data['daily_range'] > 0,
        abs(data['gap_magnitude']) / (data['daily_range'] / data['prev_close']),
        0
    )
    
    # Cross-session divergence
    data['morning_afternoon_position_divergence'] = data['morning_range_position'] - data['afternoon_range_position']
    data['gap_resolution_pattern'] = data['gap_magnitude'] * (data['afternoon_range_position'] - data['morning_range_position'])
    data['range_acceleration_divergence'] = data['morning_range_acceleration'] - data['afternoon_range_acceleration']
    
    # Volume concentration analysis
    data['total_volume'] = data['volume']
    data['morning_volume_intensity'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x.iloc[0] if len(x) == 2 else 0.5) / data['total_volume']
    data['afternoon_volume_concentration'] = (data['total_volume'] - data['morning_volume_intensity'] * data['total_volume']) / data['total_volume']
    data['volume_decay_pattern'] = data['morning_volume_intensity'] - data['afternoon_volume_concentration']
    
    # Volume efficiency components
    data['morning_volume_efficiency'] = data['morning_gap_range_efficiency'] * data['morning_volume_intensity']
    data['afternoon_volume_efficiency'] = data['afternoon_range_position'] * data['afternoon_volume_concentration']
    data['volume_efficiency_divergence'] = data['morning_volume_efficiency'] - data['afternoon_volume_efficiency']
    
    # Volume acceleration and turnover
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_acceleration'] = np.where(
        data['prev_volume'] > 0,
        ((data['volume'] - data['prev_volume']) / data['prev_volume']) - data['volume_decay_pattern'],
        0
    )
    
    data['prev_amount'] = data['amount'].shift(1)
    data['turnover_momentum'] = np.where(
        (data['prev_volume'] > 0) & (data['prev_amount'] > 0),
        ((data['volume']/data['amount'] - data['prev_volume']/data['prev_amount']) / (data['prev_volume']/data['prev_amount'])),
        0
    )
    
    data['volume_price_alignment'] = data['range_acceleration_divergence'] * data['volume_acceleration']
    
    # Momentum components
    data['morning_momentum'] = (data['first_hour_close'] - data['open']) / data['open']
    data['afternoon_momentum'] = (data['close'] - data['first_hour_close']) / data['first_hour_close']
    data['cross_session_momentum_transfer'] = data['morning_momentum'] * data['afternoon_momentum']
    
    # Range-momentum divergence patterns
    data['morning_strong_weak_afternoon'] = data['morning_range_position'] * (1 - data['afternoon_momentum'])
    data['morning_weak_strong_afternoon'] = (1 - data['morning_range_position']) * data['afternoon_momentum']
    data['gap_range_momentum_mismatch'] = data['gap_magnitude'] * (data['morning_momentum'] - data['afternoon_momentum'])
    
    # Volume-weighted momentum
    data['morning_volume_weighted_momentum'] = data['morning_momentum'] * data['morning_volume_intensity']
    data['afternoon_volume_weighted_momentum'] = data['afternoon_momentum'] * data['afternoon_volume_concentration']
    data['volume_momentum_divergence'] = data['morning_volume_weighted_momentum'] - data['afternoon_volume_weighted_momentum']
    
    # Range expansion analysis
    data['prev_morning_range_val'] = data['morning_range'].shift(1)
    data['morning_range_expansion'] = np.where(
        data['prev_morning_range_val'] > 0,
        (data['morning_range'] / data['prev_morning_range_val']) - 1,
        0
    )
    
    data['prev_daily_range'] = data['daily_range'].shift(1)
    data['afternoon_range_expansion'] = np.where(
        data['prev_daily_range'] > 0,
        (data['daily_range'] / data['prev_daily_range']) - 1,
        0
    )
    
    data['range_expansion_divergence'] = data['morning_range_expansion'] - data['afternoon_range_expansion']
    
    # Extreme volume patterns (simplified)
    data['morning_extreme_volume'] = data['morning_volume_intensity'] * data['morning_range']
    data['afternoon_extreme_volume'] = data['afternoon_volume_concentration'] * data['daily_range']
    data['extreme_volume_divergence'] = data['morning_extreme_volume'] - data['afternoon_extreme_volume']
    
    # Gap resolution extreme patterns
    data['gap_fill_efficiency'] = data['gap_magnitude'] * (1 - data['gap_persistence_ratio'])
    data['gap_continuation_strength'] = data['gap_magnitude'] * data['gap_persistence_ratio']
    data['volume_weighted_gap_resolution'] = data['gap_resolution_pattern'] * data['volume_decay_pattern']
    
    # Primary session components
    data['gap_range_efficiency_signal'] = data['morning_gap_range_efficiency'] * data['afternoon_range_acceleration']
    data['volume_weighted_divergence'] = data['volume_efficiency_divergence'] * data['volume_momentum_divergence']
    
    # Combine range-momentum divergence patterns
    data['range_momentum_divergence_patterns'] = (
        data['morning_strong_weak_afternoon'] + 
        data['morning_weak_strong_afternoon'] + 
        data['gap_range_momentum_mismatch']
    ) / 3
    
    data['cross_session_momentum_factor'] = data['cross_session_momentum_transfer'] * data['range_momentum_divergence_patterns']
    
    # Signal enhancement factors
    data['range_expansion_confirmation'] = data['range_expansion_divergence'] * data['volume_weighted_gap_resolution']
    data['extreme_behavior_adjustment'] = data['extreme_volume_divergence'] * (
        data['gap_fill_efficiency'] + data['gap_continuation_strength']
    ) / 2
    data['volume_efficiency_multiplier'] = data['volume_price_alignment'] * data['turnover_momentum']
    
    # Combine primary components
    data['primary_session_components'] = (
        data['gap_range_efficiency_signal'] + 
        data['volume_weighted_divergence'] + 
        data['cross_session_momentum_factor']
    ) / 3
    
    # Combine enhancement factors
    data['signal_enhancement_factors'] = (
        data['range_expansion_confirmation'] + 
        data['extreme_behavior_adjustment'] + 
        data['volume_efficiency_multiplier']
    ) / 3
    
    # Final alpha output
    alpha = data['primary_session_components'] * data['signal_enhancement_factors']
    
    # Clean up and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = alpha.fillna(method='ffill').fillna(0)
    
    return alpha
