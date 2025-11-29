import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic daily metrics
    data['prev_close'] = data['close'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev2_close'] = data['close'].shift(2)
    data['prev2_amount'] = data['amount'].shift(2)
    data['prev2_volume'] = data['volume'].shift(2)
    
    # Calculate daily returns and ranges
    data['daily_return'] = data['close'] - data['open']
    data['prev_daily_return'] = data['daily_return'].shift(1)
    data['daily_range'] = data['high'] - data['low']
    
    # For simplicity, we'll use first hour data as first 1/6 of trading day and last hour as last 1/6
    # In practice, these would be actual intraday data
    data['first_hour_high'] = data['high'].rolling(window=6, min_periods=1).apply(lambda x: x[:1].max() if len(x) >= 1 else np.nan)
    data['first_hour_low'] = data['low'].rolling(window=6, min_periods=1).apply(lambda x: x[:1].min() if len(x) >= 1 else np.nan)
    data['first_hour_close'] = data['close'].rolling(window=6, min_periods=1).apply(lambda x: x[0] if len(x) >= 1 else np.nan)
    data['first_hour_amount'] = data['amount'].rolling(window=6, min_periods=1).apply(lambda x: x[:1].sum() if len(x) >= 1 else np.nan)
    data['first_hour_volume'] = data['volume'].rolling(window=6, min_periods=1).apply(lambda x: x[:1].sum() if len(x) >= 1 else np.nan)
    
    data['last_hour_high'] = data['high'].rolling(window=6, min_periods=1).apply(lambda x: x[-1:].max() if len(x) >= 1 else np.nan)
    data['last_hour_low'] = data['low'].rolling(window=6, min_periods=1).apply(lambda x: x[-1:].min() if len(x) >= 1 else np.nan)
    data['last_hour_close'] = data['close'].rolling(window=6, min_periods=1).apply(lambda x: x[-1] if len(x) >= 1 else np.nan)
    data['last_hour_amount'] = data['amount'].rolling(window=6, min_periods=1).apply(lambda x: x[-1:].sum() if len(x) >= 1 else np.nan)
    data['last_hour_volume'] = data['volume'].rolling(window=6, min_periods=1).apply(lambda x: x[-1:].sum() if len(x) >= 1 else np.nan)
    
    # Midday data (middle 2/3 of day)
    data['midday_high'] = data['high'].rolling(window=6, min_periods=1).apply(lambda x: x[1:5].max() if len(x) >= 5 else np.nan)
    data['midday_low'] = data['low'].rolling(window=6, min_periods=1).apply(lambda x: x[1:5].min() if len(x) >= 5 else np.nan)
    data['midday_amount'] = data['amount'].rolling(window=6, min_periods=1).apply(lambda x: x[1:5].sum() if len(x) >= 5 else np.nan)
    data['midday_volume'] = data['volume'].rolling(window=6, min_periods=1).apply(lambda x: x[1:5].sum() if len(x) >= 5 else np.nan)
    
    # Calculate amount-weighted price efficiency
    data['amount_volume_ratio'] = data['amount'] / data['volume']
    data['first_hour_av_ratio'] = data['first_hour_amount'] / data['first_hour_volume']
    data['last_hour_av_ratio'] = data['last_hour_amount'] / data['last_hour_volume']
    data['midday_av_ratio'] = data['midday_amount'] / data['midday_volume']
    
    # Opening Range-Flow Efficiency Patterns
    # Gap Range-Flow Elasticity
    data['gap_absorption_rf'] = ((data['first_hour_high'] - data['first_hour_low']) * 
                                data['first_hour_av_ratio'] * np.sign(data['open'] - data['prev_close']))
    
    data['gap_momentum_rf'] = (np.sign(data['open'] - data['prev_close']) * 
                              (data['first_hour_close'] - data['open']) * data['first_hour_av_ratio'])
    
    data['gap_position_rf'] = ((data['open'] - data['low']) / (data['high'] - data['low']) * 
                              data['first_hour_av_ratio'])
    
    # Early Session Range-Flow Synchronization
    data['range_flow_efficiency'] = ((data['first_hour_close'] - data['open']) * 
                                    data['first_hour_av_ratio'] * np.sign(data['open'] - data['prev_close']))
    
    data['opening_range_flow_density'] = ((data['first_hour_high'] - data['first_hour_low']) * 
                                         data['first_hour_av_ratio'] * np.sign(data['first_hour_close'] - data['open']))
    
    data['opening_range_flow_quality'] = (np.sign(data['open'] - data['prev_close']) * 
                                         np.sign(data['first_hour_close'] - data['open']) * 
                                         data['first_hour_av_ratio'])
    
    # Intraday Range-Flow Transition Dynamics
    # Session Boundary Range-Flow Shifts
    morning_flow = (data['first_hour_close'] - data['open']) * data['first_hour_av_ratio']
    afternoon_flow = (data['close'] - data['last_hour_low']) * data['last_hour_av_ratio']
    data['morning_afternoon_ratio'] = afternoon_flow / morning_flow
    
    data['range_flow_efficiency_transition'] = ((data['close'] - data['open']) * data['amount_volume_ratio'] * 
                                               (data['first_hour_close'] - data['open']) * data['first_hour_av_ratio'])
    
    data['range_flow_transition_quality'] = ((data['last_hour_high'] - data['last_hour_low']) / 
                                            (data['first_hour_high'] - data['first_hour_low']) * 
                                            np.sign(data['close'] - data['open']))
    
    # Range-Flow Compression Synchronization
    data['range_flow_compression_efficiency'] = (data['daily_range'] * data['amount_volume_ratio'] * 
                                                (data['close'] - data['open']) * data['amount_volume_ratio'])
    
    data['amount_weighted_range_flow'] = (data['close'] - data['open']) * data['amount_volume_ratio'] * data['amount_volume_ratio']
    
    # Range-Flow Efficiency Concentration
    # Range-Flow Timing Effects
    morning_amount = data['first_hour_amount']
    afternoon_amount = data['last_hour_amount']
    data['range_flow_concentration_quality'] = ((morning_amount / afternoon_amount) * 
                                               (data['close'] - data['open']) * data['amount_volume_ratio'])
    
    data['range_flow_timing_mismatch'] = ((morning_amount / afternoon_amount) * 
                                         (data['close'] - data['open']) * data['amount_volume_ratio'])
    
    # Range-Flow Clustering Patterns
    data['consecutive_range_flow_expansion'] = (np.sign(data['daily_return'] - data['prev_daily_return']) * 
                                               np.sign(data['amount'] - data['prev_amount']))
    
    prev_flow = (data['prev_close'] - data['open'].shift(1)) * data['prev_amount'] / data['prev_volume']
    data['range_flow_efficiency_divergence'] = ((data['close'] - data['open']) * data['amount_volume_ratio'] * prev_flow)
    
    data['multi_scale_range_flow_efficiency'] = data['gap_momentum_rf'] * data['gap_position_rf']
    
    # Closing Session Range-Flow Validation
    # Final Hour Range-Flow Performance
    last_hour_range = data['last_hour_high'] - data['last_hour_low']
    data['closing_range_flow_intensity'] = ((data['close'] - data['last_hour_low']) / last_hour_range * 
                                           data['last_hour_av_ratio'])
    
    data['close_position_range_flow_efficiency'] = ((data['close'] - data['low']) / data['daily_range'] * 
                                                   data['amount_volume_ratio'])
    
    data['late_session_range_flow_acceleration'] = ((data['close'] - data['last_hour_low']) / last_hour_range * 
                                                   data['last_hour_av_ratio'])
    
    # Multi-Timeframe Range-Flow Memory
    # Short-Term Range-Flow Cycles
    data['intraday_range_flow_persistence'] = ((data['close'] - data['open']) * data['amount_volume_ratio'] * 
                                              np.sign(data['open'] - data['prev_close']))
    
    data['session_transition_range_flow'] = (np.sign(data['first_hour_close'] - data['open']) * 
                                            np.sign(data['close'] - data['last_hour_low']) * 
                                            data['first_hour_av_ratio'])
    
    current_flow = (data['close'] - data['open']) * data['amount_volume_ratio']
    prev_flow = (data['prev_close'] - data['open'].shift(1)) * data['prev_amount'] / data['prev_volume']
    data['range_flow_synchronization_change'] = current_flow - prev_flow
    
    # Medium-Term Range-Flow Integration
    data['previous_day_range_flow_memory'] = (data['prev_daily_return'] * 
                                             (data['close'] - data['open']) * data['amount_volume_ratio'])
    
    data['gap_range_flow_absorption'] = (abs(data['open'] - data['prev_close']) * 
                                        (data['prev_amount'] / data['prev_volume']) * 
                                        (data['close'] - data['prev_close']) * data['amount_volume_ratio'])
    
    # Composite factors
    data['opening_range_flow_factor'] = ((data['gap_absorption_rf'] + data['gap_momentum_rf'] + data['gap_position_rf']) / 3 * 
                                        data['opening_range_flow_quality'])
    
    data['transition_range_flow_factor'] = (data['morning_afternoon_ratio'] * 
                                           data['range_flow_compression_efficiency'])
    
    data['concentration_quality_factor'] = (data['range_flow_concentration_quality'] * 
                                           data['range_flow_timing_mismatch'])
    
    data['closing_validation_factor'] = (data['closing_range_flow_intensity'] * 
                                        data['late_session_range_flow_acceleration'])
    
    data['range_flow_memory_factor'] = (data['intraday_range_flow_persistence'] * 
                                       data['gap_range_flow_absorption'])
    
    # Final composite alpha
    alpha = (data['opening_range_flow_factor'] * 
             data['transition_range_flow_factor'] * 
             data['concentration_quality_factor'] * 
             data['closing_validation_factor'] * 
             data['range_flow_memory_factor'])
    
    return alpha
