import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous close
    data['prev_close'] = data['close'].shift(1)
    
    # Calculate 30-minute high/low (approximated as first 30 minutes of trading)
    # Assuming first 30 minutes is roughly 1/13 of trading day
    data['high_30min'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x.max() if len(x) == 2 else np.nan)
    data['low_30min'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x.min() if len(x) == 2 else np.nan)
    
    # Calculate previous day's high-low range
    data['prev_hl_range'] = (data['high'] - data['low']).shift(2)
    
    # Calculate volume and amount ratios (approximating 30-minute periods)
    data['volume_first_30min'] = data['volume'] * 0.08  # Approximate first 30min volume
    data['volume_last_30min'] = data['volume'] * 0.08   # Approximate last 30min volume
    data['amount_first_30min'] = data['amount'] * 0.08  # Approximate first 30min amount
    data['amount_last_30min'] = data['amount'] * 0.08   # Approximate last 30min amount
    
    # Intraday Fracture Momentum Patterns
    # Morning Fracture Momentum
    data['opening_fracture_intensity'] = ((data['open'] - data['prev_close']) / 
                                         (data['high_30min'] - data['low_30min'])) * \
                                        ((data['high_30min'] - data['low_30min']) / data['prev_hl_range'])
    
    data['early_session_momentum_fracture'] = ((data['high_30min'] - data['open']) / 
                                              (data['open'] - data['prev_close'])) * \
                                             np.sign(data['close'] - data['open'])
    
    data['fracture_persistence_ratio'] = (np.abs(data['close'] - data['open']) / 
                                         np.abs(data['high_30min'] - data['low_30min'])) * \
                                        ((data['high_30min'] - data['low_30min']) / data['prev_hl_range'])
    
    # Afternoon Fracture Reversal
    data['late_session_fracture_reversal'] = ((data['close'] - data['low_30min']) / 
                                             (data['high_30min'] - data['low_30min'])) * \
                                            np.sign(data['open'] - data['prev_close'])
    
    data['end_of_day_fracture_momentum'] = ((data['close'] - data['open']) / 
                                           (data['high_30min'] - data['low_30min'])) * \
                                          ((data['high_30min'] - data['low_30min']) / data['prev_hl_range'])
    
    data['fracture_reversal_confirmation'] = np.sign(data['close'] - data['open']) * \
                                            np.sign(data['open'] - data['prev_close']) * -1
    
    # Volume-Fracture Dynamics
    # Fracture Volume Concentration
    data['morning_volume_fracture'] = (data['volume_first_30min'] / data['volume']) * \
                                     ((data['open'] - data['prev_close']) / 
                                      (data['high_30min'] - data['low_30min']))
    
    data['afternoon_volume_fracture'] = (data['volume_last_30min'] / data['volume']) * \
                                       ((data['close'] - data['open']) / 
                                        (data['high_30min'] - data['low_30min']))
    
    data['volume_fracture_divergence'] = data['morning_volume_fracture'] - data['afternoon_volume_fracture']
    
    # Amount-Fracture Alignment
    data['fracture_amount_intensity'] = (data['amount_first_30min'] / data['amount']) * \
                                       (np.abs(data['open'] - data['prev_close']) / 
                                        (data['high_30min'] - data['low_30min']))
    
    data['amount_fracture_persistence'] = (data['amount_last_30min'] / data['amount']) * \
                                         (np.abs(data['close'] - data['open']) / 
                                          (data['high_30min'] - data['low_30min']))
    
    data['amount_fracture_confirmation'] = np.sign(data['open'] - data['prev_close']) * \
                                          (data['amount_first_30min'] / data['amount'])
    
    # Multi-session Fracture Patterns
    # Fracture Momentum Persistence
    data['open_close_sign'] = np.sign(data['open'] - data['prev_close'])
    data['prev_open_close_sign'] = data['open_close_sign'].shift(1)
    
    # Consecutive Fracture Days
    data['consecutive_fracture_days'] = 0
    for i in range(2, len(data)):
        count = 0
        for j in range(max(0, i-5), i):
            if data['open_close_sign'].iloc[j] == data['prev_open_close_sign'].iloc[j]:
                count += 1
        data.loc[data.index[i], 'consecutive_fracture_days'] = count
    
    data['fracture_momentum_ratio'] = (np.abs(data['open'] - data['prev_close']) / 
                                      np.abs(data['open'].shift(1) - data['prev_close'].shift(1))) * \
                                     ((data['high_30min'] - data['low_30min']) / data['prev_hl_range'])
    
    data['fracture_direction_consistency'] = np.sign(data['open'] - data['prev_close']) * \
                                            np.sign(data['close'] - data['open']) * -1
    
    # Fracture Gap Integration
    data['fracture_gap_magnitude'] = (np.abs(data['open'] - data['prev_close']) / 
                                     (data['high_30min'] - data['low_30min'])) * \
                                    ((data['high_30min'] - data['low_30min']) / data['prev_hl_range'])
    
    data['gap_fracture_efficiency'] = ((data['close'] - np.minimum(data['open'], data['prev_close'])) / 
                                      np.abs(data['open'] - data['prev_close'])) * \
                                     np.sign(data['close'] - data['open'])
    
    data['multi_day_fracture_gap'] = (np.abs(data['open'] - data['prev_close']) / 
                                     np.abs(data['open'].shift(1) - data['prev_close'].shift(1))) * \
                                    ((data['high_30min'] - data['low_30min']) / data['prev_hl_range'])
    
    # Combine all factors with weights
    factors = [
        'opening_fracture_intensity', 'early_session_momentum_fracture', 'fracture_persistence_ratio',
        'late_session_fracture_reversal', 'end_of_day_fracture_momentum', 'fracture_reversal_confirmation',
        'morning_volume_fracture', 'afternoon_volume_fracture', 'volume_fracture_divergence',
        'fracture_amount_intensity', 'amount_fracture_persistence', 'amount_fracture_confirmation',
        'consecutive_fracture_days', 'fracture_momentum_ratio', 'fracture_direction_consistency',
        'fracture_gap_magnitude', 'gap_fracture_efficiency', 'multi_day_fracture_gap'
    ]
    
    # Calculate composite factor (equal weights for demonstration)
    composite_factor = data[factors].mean(axis=1, skipna=True)
    
    return composite_factor
