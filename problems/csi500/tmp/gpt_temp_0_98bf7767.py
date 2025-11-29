import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['range'] = data['high'] - data['low']
    data['midday'] = (data['high'] + data['low']) / 2
    
    # AM Momentum calculations
    data['am_up_momentum'] = data['high'] - data['open']
    data['am_down_momentum'] = data['open'] - data['low']
    data['am_net_momentum'] = data['am_up_momentum'] - data['am_down_momentum']
    data['am_momentum_direction'] = np.sign(data['am_net_momentum'])
    
    # PM Momentum calculations
    data['pm_up_momentum'] = np.where(data['close'] > data['midday'], data['close'] - data['midday'], 0)
    data['pm_down_momentum'] = np.where(data['close'] < data['midday'], data['midday'] - data['close'], 0)
    data['pm_net_momentum'] = data['pm_up_momentum'] - data['pm_down_momentum']
    data['pm_momentum_direction'] = np.sign(data['pm_net_momentum'])
    
    # Momentum Direction Divergence
    data['momentum_direction_divergence'] = np.where(
        data['am_momentum_direction'] == data['pm_momentum_direction'], 
        np.abs(data['am_net_momentum'] + data['pm_net_momentum']),
        -np.abs(data['am_net_momentum'] - data['pm_net_momentum'])
    )
    
    # Price Velocity calculations
    data['am_velocity'] = data['am_net_momentum'] / (data['range'] + 1e-8)
    data['pm_velocity'] = data['pm_net_momentum'] / (data['range'] + 1e-8)
    data['velocity_acceleration'] = data['pm_velocity'] - data['am_velocity']
    
    # Momentum-Velocity Divergence
    data['momentum_velocity_divergence'] = np.where(
        np.sign(data['am_net_momentum']) == np.sign(data['velocity_acceleration']),
        np.abs(data['am_net_momentum'] * data['velocity_acceleration']),
        -np.abs(data['am_net_momentum'] * data['velocity_acceleration'])
    )
    
    # Multi-session momentum persistence (3-day rolling)
    data['momentum_persistence_3d'] = (
        data['am_momentum_direction'].rolling(window=3).sum() + 
        data['pm_momentum_direction'].rolling(window=3).sum()
    ) / 6.0
    
    # Momentum Efficiency Ratio
    data['momentum_efficiency'] = (
        (np.abs(data['am_net_momentum']) + np.abs(data['pm_net_momentum'])) / 
        (data['range'] + 1e-8)
    )
    
    # Momentum Quality Score
    data['momentum_quality'] = (
        data['momentum_persistence_3d'] * data['momentum_efficiency'] * 
        data['momentum_direction_divergence']
    )
    
    # Volume calculations
    data['am_volume_momentum'] = data['volume'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.mean() + 1e-8), raw=False
    )
    data['pm_volume_momentum'] = data['volume'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[-2]) / (x.mean() + 1e-8), raw=False
    )
    
    # Volume-Price Efficiency
    data['volume_price_efficiency'] = (
        (np.abs(data['am_net_momentum']) + np.abs(data['pm_net_momentum'])) / 
        (data['volume'] + 1e-8)
    )
    
    # Amount calculations
    data['amount_momentum_correlation'] = data['amount'].rolling(window=5).corr(
        other=(np.abs(data['am_net_momentum']) + np.abs(data['pm_net_momentum']))
    )
    data['amount_volume_divergence'] = data['amount'] / (data['volume'] + 1e-8)
    
    # Volume-Price Asymmetry Factor
    data['volume_price_asymmetry'] = (
        data['volume_price_efficiency'] * data['amount_momentum_correlation'] * 
        data['amount_volume_divergence'] * np.sign(data['am_net_momentum'])
    )
    
    # Combine Momentum and Efficiency Factors
    data['momentum_efficiency_divergence'] = (
        data['momentum_quality'] * data['volume_price_asymmetry']
    )
    
    # Multi-timeframe context (5-day efficiency patterns)
    data['efficiency_trend_5d'] = data['volume_price_efficiency'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.std() + 1e-8), raw=False
    )
    
    # Price Range Utilization
    data['range_momentum_efficiency'] = (
        (np.abs(data['am_net_momentum']) + np.abs(data['pm_net_momentum'])) / 
        (data['range'] + 1e-8)
    )
    data['range_volume_alignment'] = data['volume'] / (data['range'] + 1e-8)
    
    # Range-Adjusted Composite
    data['range_adjusted_composite'] = (
        data['momentum_efficiency_divergence'] * data['range_momentum_efficiency'] * 
        data['range_volume_alignment']
    )
    
    # Momentum Acceleration Patterns
    data['am_momentum_acceleration'] = data['am_net_momentum'].diff()
    data['pm_momentum_acceleration'] = data['pm_net_momentum'].diff()
    data['acceleration_divergence'] = data['pm_momentum_acceleration'] - data['am_momentum_acceleration']
    
    # Acceleration Quality
    data['acceleration_persistence'] = (
        np.sign(data['am_momentum_acceleration']).rolling(window=3).sum() / 3.0
    )
    
    # Exhaustion Signals
    data['velocity_deceleration'] = -data['velocity_acceleration'].diff()
    data['volume_efficiency_deterioration'] = -data['volume_price_efficiency'].diff()
    
    # Exhaustion-Confirmation Strength
    data['exhaustion_strength'] = (
        data['velocity_deceleration'] + data['volume_efficiency_deterioration']
    ) / 2.0
    
    # Acceleration-Exhaustion Factor
    data['acceleration_exhaustion_factor'] = (
        data['acceleration_persistence'] * data['exhaustion_strength'] * 
        np.sign(data['am_net_momentum'])
    )
    
    # Final Composite Factor
    data['composite_factor'] = (
        data['range_adjusted_composite'] * data['acceleration_exhaustion_factor']
    )
    
    # Apply dynamic scaling with hyperbolic tangent
    data['final_factor'] = np.tanh(data['composite_factor'] / (data['composite_factor'].std() + 1e-8))
    
    # Apply logarithmic scaling for extreme values
    extreme_mask = np.abs(data['final_factor']) > 2
    data.loc[extreme_mask, 'final_factor'] = np.sign(data.loc[extreme_mask, 'final_factor']) * np.log1p(
        np.abs(data.loc[extreme_mask, 'final_factor'])
    )
    
    return data['final_factor']
