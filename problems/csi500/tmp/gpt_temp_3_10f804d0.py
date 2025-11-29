import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price-Volume Divergence Signal
    # Price Momentum Acceleration
    data['price_velocity'] = (data['close'] - data['open']) / data['open']
    data['price_acceleration'] = data['price_velocity'] - data['price_velocity'].shift(1)
    data['momentum_strength'] = np.abs(data['price_acceleration']) * np.sign(data['price_acceleration'])
    
    # Volume-Price Alignment
    data['volume_weighted_price_change'] = (data['close'] - data['open']) * data['volume']
    data['avg_volume_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['expected_alignment'] = np.abs(data['close'] - data['open']) * data['avg_volume_5d']
    data['divergence_ratio'] = data['volume_weighted_price_change'] / data['expected_alignment'].replace(0, np.nan)
    
    # Intraday Range Dynamics
    # Range Efficiency Metric
    data['actual_movement'] = np.abs(data['close'] - data['open'])
    data['available_range'] = data['high'] - data['low']
    data['range_utilization'] = data['actual_movement'] / data['available_range'].replace(0, np.nan)
    
    # Range Expansion Patterns
    data['daily_range'] = data['high'] - data['low']
    data['range_change'] = data['daily_range'] - data['daily_range'].shift(1)
    data['range_momentum'] = np.sign(data['range_change']) * np.abs(data['range_change'])
    data['range_dynamics'] = data['range_utilization'] * data['range_momentum']
    
    # Volume Flow Characteristics
    # Simplified Volume Distribution Profile (using daily aggregates as proxy)
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).sum()
    data['volume_timing'] = data['volume_concentration'] * data['volume_concentration']
    
    # Volume Persistence
    data['volume_autocorr'] = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    data['volume_std_5d'] = data['volume'].rolling(window=5, min_periods=1).std()
    data['volume_stability'] = 1 / (data['volume_std_5d'] / data['avg_volume_5d'].replace(0, np.nan))
    data['volume_persistence'] = data['volume_autocorr'] * data['volume_stability']
    
    # Price Level Context
    # Relative Position Strength
    data['position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['closing_bias'] = np.abs(data['position'] - 0.5) * np.sign(data['position'] - 0.5)
    data['level_momentum'] = data['closing_bias'] - data['closing_bias'].shift(1)
    
    # Support/Resistance Dynamics
    data['distance_to_high'] = data['high'] - data['close']
    data['distance_to_low'] = data['close'] - data['low']
    data['distance_to_extremes'] = np.minimum(data['distance_to_high'], data['distance_to_low']) / data['available_range'].replace(0, np.nan)
    data['boundary_pressure'] = 1 / data['distance_to_extremes'].replace(0, np.nan)
    data['level_significance'] = data['boundary_pressure'] * data['level_momentum']
    
    # Generate Final Alpha Factor
    # Combine Core Divergence Components
    data['primary_signal'] = data['divergence_ratio'] * data['range_dynamics']
    data['volume_enhanced'] = data['primary_signal'] * data['volume_timing']
    data['level_enhanced'] = data['volume_enhanced'] * data['level_significance']
    
    # Apply Momentum Acceleration Filter
    data['acceleration_filtered'] = data['level_enhanced'] * data['momentum_strength']
    data['final_factor'] = data['acceleration_filtered'] * data['volume_persistence']
    
    # Fill NaN values with 0
    data['final_factor'] = data['final_factor'].fillna(0)
    
    return data['final_factor']
