import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic daily metrics
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_open'] = data['open'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Daily range calculations
    data['daily_range'] = data['high'] - data['low']
    data['prev_daily_range'] = data['prev_high'] - data['prev_low']
    
    # Opening Session Flow-Momentum Quality
    # Gap-Flow-Momentum Alignment
    data['raw_gap_size'] = data['open'] - data['prev_close']
    data['gap_direction'] = np.sign(data['raw_gap_size'])
    
    # For simplicity, using first hour as first 25% of trading day (approximation)
    data['first_hour_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x.max() if len(x) == 2 else x.iloc[0])
    data['first_hour_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x.min() if len(x) == 2 else x.iloc[0])
    data['first_hour_range'] = data['first_hour_high'] - data['first_hour_low']
    
    # Approximate first hour metrics using rolling calculations
    data['first_hour_volume_ratio'] = data['volume'].rolling(window=3, min_periods=1).apply(lambda x: x.iloc[0] / x.sum() if len(x) == 3 else 0.33)
    data['gap_filling_behavior'] = np.where(data['raw_gap_size'] != 0, 
                                          (data['close'] - data['open']) / data['raw_gap_size'], 0)
    
    # Flow Distribution Efficiency
    data['early_session_flow_concentration'] = data['first_hour_volume_ratio']
    data['range_flow_efficiency'] = np.where(data['amount'] != 0, data['first_hour_range'] / data['amount'], 0)
    data['opening_flow_quality'] = np.where(data['amount'] != 0, np.abs(data['close'] - data['open']) / data['amount'], 0)
    
    # Flow Rejection & Support Dynamics
    data['upper_flow_rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) - (data['prev_high'] - np.maximum(data['prev_open'], data['prev_close']))
    data['lower_flow_support'] = (np.minimum(data['open'], data['close']) - data['low']) - (np.minimum(data['prev_open'], data['prev_close']) - data['prev_low'])
    data['net_flow_pressure'] = data['upper_flow_rejection'] - data['lower_flow_support']
    
    # Intraday Compression & Breakout Dynamics
    data['daily_range_flow_ratio'] = np.where(data['prev_daily_range'] != 0, data['daily_range'] / data['prev_daily_range'], 1)
    data['flow_compression'] = np.where(data['prev_amount'] != 0, data['amount'] / data['prev_amount'], 1)
    data['range_contraction_analysis'] = np.where(data['prev_daily_range'] != 0, data['daily_range'] / data['prev_daily_range'], 1)
    data['flow_compression_intensity'] = data['daily_range_flow_ratio'] * data['flow_compression']
    
    # Breakout Quality Assessment
    data['flow_breakout_strength'] = (data['high'] - data['prev_high']) - (data['prev_low'] - data['low'])
    data['amount_surge_intensity'] = np.where(data['prev_amount'] != 0, data['amount'] / data['prev_amount'], 1)
    data['flow_confirmed_breakout'] = data['flow_breakout_strength'] * np.sign(data['amount'] - data['prev_amount'])
    data['breakout_acceleration_efficiency'] = np.where(data['daily_range'] != 0, (data['close'] - data['open']) / data['daily_range'], 0)
    
    # Flow Fragmentation Timing
    data['late_session_volume_ratio'] = data['volume'].rolling(window=3, min_periods=1).apply(lambda x: x.iloc[-1] / x.sum() if len(x) == 3 else 0.33)
    data['flow_fragmentation_score'] = data['early_session_flow_concentration'] - data['late_session_volume_ratio']
    data['flow_price_fragmentation_mismatch'] = data['flow_fragmentation_score'] * data['breakout_acceleration_efficiency']
    
    # Price-Level Flow & Momentum Behavior (simplified approximations)
    data['round_number_flow_patterns'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['flow_direction_consistency'] = data['close'].rolling(window=3, min_periods=1).apply(lambda x: len(set(np.sign(np.diff(x)))) if len(x) == 3 else 1)
    data['flow_intensity_reduction'] = data['amount'] / data['amount'].rolling(window=5, min_periods=1).mean()
    
    # Temporal Flow-Momentum Asymmetry
    data['session_flow_divergence'] = data['open'].rolling(window=2, min_periods=1).apply(lambda x: np.sign(x.iloc[1] - x.iloc[0]) if len(x) == 2 else 0) * data['close'].rolling(window=2, min_periods=1).apply(lambda x: np.sign(x.iloc[1] - x.iloc[0]) if len(x) == 2 else 0)
    data['hourly_flow_momentum_quality'] = data['amount'] * data['flow_direction_consistency']
    data['session_end_flow_consolidation'] = np.where(data['daily_range'] != 0, (data['close'] - data['low']) / data['daily_range'], 0.5)
    
    # Composite Flow-Momentum Efficiency Factors
    data['opening_component'] = data['gap_direction'] * data['opening_flow_quality'] * data['net_flow_pressure']
    data['breakout_component'] = data['flow_compression_intensity'] * data['flow_confirmed_breakout'] * data['breakout_acceleration_efficiency']
    data['price_level_component'] = data['round_number_flow_patterns'] * data['flow_direction_consistency'] * data['flow_intensity_reduction']
    data['temporal_component'] = data['session_flow_divergence'] * data['hourly_flow_momentum_quality'] * data['session_end_flow_consolidation']
    
    # Final Alpha Factor
    data['alpha_factor'] = data['opening_component'] * data['breakout_component'] * data['price_level_component'] * data['temporal_component']
    
    # Clean up and return
    result = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
