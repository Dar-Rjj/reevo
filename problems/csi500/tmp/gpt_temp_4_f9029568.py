import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous values
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_range'] = data['prev_high'] - data['prev_low']
    
    # Avoid division by zero
    data['prev_range'] = data['prev_range'].replace(0, np.nan)
    data['high_low'] = data['high'] - data['low']
    data['high_low'] = data['high_low'].replace(0, np.nan)
    
    # Intraday Price Momentum Dislocation
    # Opening Momentum Dynamics
    data['opening_gap_momentum'] = (data['open'] - data['prev_close']) * data['volume']
    data['opening_range_capture'] = ((data['high'] - data['open']) / data['high_low']) * data['amount']
    data['opening_momentum_efficiency'] = data['opening_gap_momentum'] * data['opening_range_capture']
    
    # Closing Momentum Dynamics
    data['closing_range_momentum'] = ((data['close'] - data['low']) / data['high_low']) * data['volume']
    data['closing_gap_formation'] = (data['close'] - data['open']) * data['amount']
    data['closing_momentum_efficiency'] = data['closing_range_momentum'] * data['closing_gap_formation']
    
    # Volume-Price Divergence Patterns
    # Volume Acceleration Signals
    data['volume_momentum'] = (data['volume'] / data['prev_volume']) * (data['close'] - data['open'])
    data['price_volume_divergence'] = data['high_low'] * (data['volume'] - data['prev_volume'])
    data['acceleration_efficiency'] = data['volume_momentum'] * data['price_volume_divergence']
    
    # Session Volume Distribution (simplified - using daily data only)
    data['early_session_dominance'] = (data['high'] - data['open']) * 0.4  # Assume 40% volume in first hour
    data['late_session_dominance'] = (data['close'] - data['low']) * 0.3   # Assume 30% volume in last hour
    data['volume_session_imbalance'] = data['early_session_dominance'] - data['late_session_dominance']
    
    # Price Range Boundary Behavior
    # Range Expansion Dynamics
    data['range_expansion_signal'] = (data['high_low'] / data['prev_range']) * data['volume']
    data['boundary_breakout_efficiency'] = ((data['close'] - data['prev_close']) / data['high_low']) * data['amount']
    data['range_momentum'] = data['range_expansion_signal'] * data['boundary_breakout_efficiency']
    
    # Micro-Range Patterns
    data['intraday_range_compression'] = (data['high_low'] / data['high_low'].rolling(5).mean()) * data['volume']
    
    # Calculate range persistence (simplified)
    data['range_direction'] = np.where(data['high_low'] > data['prev_range'], 1, -1)
    data['range_persistence'] = data['range_direction'].groupby(data.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1
    ).reset_index(level=0, drop=True)
    data['range_persistence_signal'] = data['range_persistence'] * data['range_momentum']
    data['micro_structure_efficiency'] = data['intraday_range_compression'] * data['range_persistence_signal']
    
    # Momentum Regime Detection
    # Trend Identification
    data['upward_trend_confirmation'] = (data['close'] - data['open']) * data['high_low'] * data['volume']
    data['downward_trend_confirmation'] = (data['open'] - data['close']) * data['high_low'] * data['amount']
    data['net_trend_signal'] = data['upward_trend_confirmation'] - data['downward_trend_confirmation']
    
    # Regime Persistence Metrics
    data['trend_direction'] = np.where(data['net_trend_signal'] > 0, 1, -1)
    data['trend_persistence'] = data['trend_direction'].groupby(data.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1
    ).reset_index(level=0, drop=True)
    data['trend_duration_factor'] = data['trend_persistence'] * data['net_trend_signal']
    data['volume_regime_alignment'] = data['volume'] * data['net_trend_signal']
    data['regime_strength'] = data['trend_duration_factor'] * data['volume_regime_alignment']
    
    # Price-Volume Cointegration
    # Cointegration Signals
    data['price_volume_correlation'] = (data['close'] - data['open']) * data['volume']
    data['volume_price_efficiency'] = data['volume'] * (data['high_low'] / data['prev_range'])
    data['cointegration_strength'] = data['price_volume_correlation'] * data['volume_price_efficiency']
    
    # Temporal Cointegration Patterns (simplified)
    data['early_session_cointegration'] = (data['open'] - data['prev_close']) * data['volume'] * 0.4
    data['late_session_cointegration'] = (data['close'] - data['open']) * data['volume'] * 0.3
    data['session_cointegration_divergence'] = data['early_session_cointegration'] - data['late_session_cointegration']
    
    # Momentum Asymmetry Detection
    # Directional Momentum Asymmetry
    data['upward_momentum_bias'] = (data['high'] - data['open']) * data['volume']
    data['downward_momentum_bias'] = (data['open'] - data['low']) * data['amount']
    data['momentum_asymmetry_signal'] = data['upward_momentum_bias'] - data['downward_momentum_bias']
    
    # Volume-Weighted Asymmetry
    data['volume_asymmetry'] = (data['volume'] * 0.5 - data['volume'] * 0.5) * (data['close'] - data['open'])
    data['price_asymmetry_efficiency'] = ((data['high'] - data['open']) - (data['close'] - data['low'])) * data['volume']
    data['asymmetry_strength'] = data['volume_asymmetry'] * data['price_asymmetry_efficiency']
    
    # Micro-Structure Inefficiency
    # Price Discovery Patterns
    data['opening_price_discovery'] = (data['high'] - data['open']) * (data['open'] - data['low']) * data['volume']
    data['closing_price_discovery'] = (data['close'] - data['low']) * (data['high'] - data['close']) * data['amount']
    data['discovery_efficiency'] = data['opening_price_discovery'] * data['closing_price_discovery']
    
    # Volume Concentration Effects
    data['volume_spike_impact'] = data['volume'] * ((data['close'] - data['prev_close']) / data['high_low'])
    data['volume_drought_effect'] = -data['volume'] * (data['high_low'] / data['prev_range'])
    data['volume_concentration_signal'] = data['volume_spike_impact'] * data['volume_drought_effect']
    
    # Composite Asymmetry Alpha
    data['momentum_efficiency_factor'] = data['opening_momentum_efficiency'] * data['closing_momentum_efficiency']
    data['volume_divergence_factor'] = data['acceleration_efficiency'] * data['volume_session_imbalance']
    data['range_dynamics_factor'] = data['range_momentum'] * data['micro_structure_efficiency']
    data['regime_strength_factor'] = data['regime_strength'] * data['net_trend_signal']
    data['cointegration_factor'] = data['cointegration_strength'] * data['session_cointegration_divergence']
    data['asymmetry_detection_factor'] = data['momentum_asymmetry_signal'] * data['asymmetry_strength']
    data['micro_structure_factor'] = data['discovery_efficiency'] * data['volume_concentration_signal']
    
    # Final Alpha
    data['final_alpha'] = (data['momentum_efficiency_factor'] * 
                          data['volume_divergence_factor'] * 
                          data['range_dynamics_factor'] * 
                          data['regime_strength_factor'] * 
                          data['cointegration_factor'] * 
                          data['asymmetry_detection_factor'] * 
                          data['micro_structure_factor'])
    
    # Return the final alpha series
    return data['final_alpha']
