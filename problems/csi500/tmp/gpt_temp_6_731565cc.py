import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['intraday_move'] = data['close'] - data['open']
    data['prev_close'] = data['close'].shift(1)
    data['daily_range'] = data['high'] - data['low']
    data['prev_range'] = data['daily_range'].shift(1)
    data['gap'] = data['open'] - data['prev_close']
    
    # Trend-Gap Interaction Analysis
    # Gap Rejection Momentum
    gap_magnitude = np.abs(data['gap'])
    gap_magnitude = gap_magnitude.replace(0, np.nan)  # Avoid division by zero
    data['gap_rejection_momentum'] = (data['intraday_move'] / gap_magnitude) * np.sign(data['intraday_move'])
    
    # Trend Persistence
    data['intraday_direction'] = np.sign(data['intraday_move'])
    data['prev_intraday_direction'] = data['intraday_direction'].shift(1)
    data['consecutive_moves'] = 0
    
    # Calculate consecutive same-direction moves
    for i in range(1, len(data)):
        if data['intraday_direction'].iloc[i] == data['prev_intraday_direction'].iloc[i]:
            data.loc[data.index[i], 'consecutive_moves'] = data['consecutive_moves'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'consecutive_moves'] = 1
    
    range_non_zero = data['daily_range'].replace(0, np.nan)
    data['trend_persistence'] = data['consecutive_moves'] * (data['intraday_move'] / range_non_zero)
    
    # Gap-Trend Alignment
    gap_sign = np.sign(data['gap'])
    intraday_sign = np.sign(data['intraday_move'])
    data['gap_trend_alignment'] = gap_sign * intraday_sign * (np.abs(data['intraday_move']) / range_non_zero)
    
    # Range Efficiency Dynamics
    # Movement Efficiency
    data['movement_efficiency'] = np.abs(data['intraday_move']) / range_non_zero
    
    # Range Momentum
    prev_range_non_zero = data['prev_range'].replace(0, np.nan)
    data['range_momentum'] = data['daily_range'] / prev_range_non_zero
    
    # Efficiency Acceleration
    data['prev_movement_efficiency'] = data['movement_efficiency'].shift(1)
    prev_eff_non_zero = data['prev_movement_efficiency'].replace(0, np.nan)
    data['efficiency_acceleration'] = (data['movement_efficiency'] - data['prev_movement_efficiency']) / prev_eff_non_zero
    
    # Volume-Price Confirmation
    # Volume Direction
    data['prev_volume'] = data['volume'].shift(1)
    prev_vol_non_zero = data['prev_volume'].replace(0, np.nan)
    data['price_change'] = data['close'] - data['prev_close']
    data['volume_direction'] = (data['volume'] / prev_vol_non_zero) * np.sign(data['price_change'])
    
    # Net Rejection Volume
    high_close_volume = data['volume'] * (data['high'] - data['close'])
    close_low_volume = data['volume'] * (data['close'] - data['low'])
    denominator = high_close_volume + close_low_volume
    denominator = denominator.replace(0, np.nan)
    data['net_rejection_volume'] = (high_close_volume - close_low_volume) / denominator
    
    # Volume-Efficiency Timing
    data['volume_efficiency_timing'] = data['net_rejection_volume'] * data['movement_efficiency'] * np.sign(data['intraday_move'])
    
    # Signal Integration
    # Core Factor
    data['core_factor'] = data['gap_rejection_momentum'] * data['range_momentum'] * data['movement_efficiency']
    
    # Confirmation Layer
    data['confirmation_layer'] = data['volume_direction'] * data['trend_persistence'] * data['efficiency_acceleration']
    
    # Final Signal
    data['final_signal'] = data['core_factor'] * data['confirmation_layer'] * data['volume_efficiency_timing']
    
    # Return the final factor values
    return data['final_signal']
