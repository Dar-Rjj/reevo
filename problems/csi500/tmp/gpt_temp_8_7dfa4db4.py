import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Pre-Session Volatility Momentum
    # Overnight Gap Momentum Strength
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Previous day range
    data['prev_range'] = (data['high'].shift(1) - data['low'].shift(1)) / data['prev_close']
    data['gap_relative_range'] = data['overnight_gap'].abs() / (data['prev_range'] + 1e-8)
    
    # Range Expansion Quality
    data['current_range'] = (data['high'] - data['low']) / data['open']
    data['prev_day_range'] = data['prev_range']
    data['range_expansion'] = data['current_range'] / (data['prev_day_range'] + 1e-8)
    
    # Directional range movement
    data['price_movement'] = (data['close'] - data['open']) / data['open']
    data['directional_range'] = data['range_expansion'] * np.sign(data['price_movement'])
    
    # Breakout Efficiency Patterns
    # Dynamic Breakout Threshold
    data['breakout_threshold'] = data['prev_range'] * 0.6
    data['gap_adjusted_threshold'] = data['breakout_threshold'] * (1 + data['overnight_gap'].abs())
    
    # Upward Breakout Momentum
    upward_breakout = data['high'] > data['open'] * (1 + data['gap_adjusted_threshold'])
    data['upward_breakout_momentum'] = 0
    mask_up = upward_breakout & (data['open'] > 0)
    data.loc[mask_up, 'upward_breakout_momentum'] = (
        (data['high'] - data['open']) / (data['open'] * data['gap_adjusted_threshold'])
    )
    data['close_position_up'] = (data['close'] - data['open']) / (data['high'] - data['open'] + 1e-8)
    
    # Downward Breakout Momentum
    downward_breakout = data['low'] < data['open'] * (1 - data['gap_adjusted_threshold'])
    data['downward_breakout_momentum'] = 0
    mask_down = downward_breakout & (data['open'] > 0)
    data.loc[mask_down, 'downward_breakout_momentum'] = (
        (data['low'] - data['open']) / (data['open'] * data['gap_adjusted_threshold'])
    )
    data['close_position_down'] = (data['close'] - data['open']) / (data['low'] - data['open'] + 1e-8)
    
    # Volume-Amount Synchronization
    # Large trade concentration
    data['large_trade_concentration'] = data['amount'] / (data['volume'] + 1e-8)
    data['prev_large_trade_concentration'] = data['large_trade_concentration'].shift(1)
    data['amount_momentum'] = (
        data['large_trade_concentration'] / (data['prev_large_trade_concentration'] + 1e-8) - 1
    )
    
    # Amount efficiency during breakout periods
    breakout_period = upward_breakout | downward_breakout
    data['price_change_per_amount'] = 0
    mask_breakout = breakout_period & (data['amount'] > 0)
    data.loc[mask_breakout, 'price_change_per_amount'] = (
        (data['close'] - data['open']) / data['amount']
    )
    
    # Volume-amount breakout confirmation
    data['volume_amount_ratio'] = data['volume'] / (data['amount'] + 1e-8)
    data['volume_amount_confirmation'] = data['volume_amount_ratio'] * data['price_movement']
    
    # Composite Efficiency Ranking
    # Breakout efficiency components
    data['breakout_efficiency'] = (
        data['upward_breakout_momentum'].fillna(0) - 
        data['downward_breakout_momentum'].fillna(0)
    )
    
    # Volume-amount synchronization score
    data['volume_amount_score'] = (
        data['amount_momentum'].fillna(0) * 
        data['volume_amount_confirmation'].fillna(0)
    )
    
    # Gap-range alignment
    data['gap_range_alignment'] = (
        data['overnight_gap'].fillna(0) * 
        data['directional_range'].fillna(0)
    )
    
    # Multi-day consistency measures (using rolling windows)
    data['breakout_momentum_3d'] = data['breakout_efficiency'].rolling(window=3, min_periods=1).mean()
    data['volume_amount_consistency_3d'] = data['volume_amount_score'].rolling(window=3, min_periods=1).std()
    data['gap_alignment_3d'] = data['gap_range_alignment'].rolling(window=3, min_periods=1).mean()
    
    # Final composite factor
    data['composite_factor'] = (
        data['breakout_efficiency'].fillna(0) * 0.4 +
        data['volume_amount_score'].fillna(0) * 0.3 +
        data['gap_range_alignment'].fillna(0) * 0.2 +
        data['breakout_momentum_3d'].fillna(0) * 0.1
    ) / (data['volume_amount_consistency_3d'].fillna(1) + 1e-8)
    
    # Cross-sectional ranking (z-score normalization)
    data['factor_rank'] = data.groupby(data.index)['composite_factor'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    return data['factor_rank']
