import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate daily true range
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['daily_true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Assume first hour data (9:30-10:30) and afternoon data (13:00-16:00)
    # For simplicity, we'll use opening hour (first 1/6.5 of trading day) and afternoon (last 3/6.5)
    df['first_hour_high'] = df['high'].rolling(window=2, min_periods=1).apply(lambda x: x.max() if len(x) == 2 else x[0])
    df['first_hour_low'] = df['low'].rolling(window=2, min_periods=1).apply(lambda x: x.min() if len(x) == 2 else x[0])
    df['first_hour_close'] = df['close'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else x[0])
    
    df['afternoon_high'] = df['high'].rolling(window=3, min_periods=1).apply(lambda x: x.max() if len(x) == 3 else x[-1])
    df['afternoon_low'] = df['low'].rolling(window=3, min_periods=1).apply(lambda x: x.min() if len(x) == 3 else x[-1])
    
    # Multi-scale Intraday Range Dynamics
    df['first_hour_range'] = df['first_hour_high'] - df['first_hour_low']
    df['morning_fractal_ratio'] = df['first_hour_range'] / df['daily_true_range']
    
    df['afternoon_range'] = df['afternoon_high'] - df['afternoon_low']
    df['afternoon_persistence_ratio'] = df['afternoon_range'] / df['daily_true_range']
    
    # Breakout Momentum with Intraday Memory
    df['morning_high_breakout'] = df['first_hour_high'] - df['prev_high']
    df['morning_low_breakdown'] = df['prev_low'] - df['first_hour_low']
    df['net_morning_breakout'] = df['morning_high_breakout'] - df['morning_low_breakdown']
    
    df['afternoon_high_breakout'] = df['afternoon_high'] - df['first_hour_high']
    df['afternoon_low_breakdown'] = df['first_hour_low'] - df['afternoon_low']
    df['net_afternoon_breakout'] = df['afternoon_high_breakout'] - df['afternoon_low_breakdown']
    
    # Volatility-Cluster Synchronized Volume
    df['prev_volume'] = df['volume'].shift(1)
    df['volume_cluster_signal'] = np.sign(df['volume'] - df['prev_volume']) * np.sign(
        (df['high'] - df['low']) - (df['prev_high'] - df['prev_low'])
    )
    
    df['amount_5d_range'] = df['amount'].rolling(window=5, min_periods=1).apply(lambda x: x.max() - x.min())
    df['amount_concentration'] = df['amount'] / df['amount_5d_range'].replace(0, 1)
    df['amount_efficiency'] = df['amount'] / df['daily_true_range'].replace(0, 1)
    
    # Intraday Momentum Decay Patterns
    df['opening_hour_return'] = (df['first_hour_close'] - df['open']) / df['open'].replace(0, 1)
    df['morning_range_utilization'] = df['first_hour_range'] / (df['high'] - df['low']).replace(0, 1)
    df['opening_momentum_quality'] = df['opening_hour_return'] * df['morning_range_utilization']
    
    df['afternoon_return'] = (df['close'] - df['first_hour_close']) / df['first_hour_close'].replace(0, 1)
    df['range_persistence_efficiency'] = df['afternoon_range'] / df['first_hour_range'].replace(0, 1)
    df['momentum_decay_factor'] = df['afternoon_return'] * df['range_persistence_efficiency']
    
    # Fractal Breakout Synchronization
    df['morning_afternoon_consistency'] = np.sign(df['morning_fractal_ratio']) * np.sign(df['afternoon_persistence_ratio'])
    df['breakout_memory_effect'] = np.sign(df['net_morning_breakout']) * np.sign(df['net_afternoon_breakout'])
    df['fractal_synchronization'] = df['morning_afternoon_consistency'] * df['breakout_memory_effect']
    
    df['range_volume_correlation'] = np.sign(
        (df['high'] - df['low']) - (df['prev_high'] - df['prev_low'])
    ) * np.sign(df['volume'] - df['prev_volume'])
    df['amount_cluster_alignment'] = df['amount_concentration'] * df['amount_efficiency']
    df['volume_synchronization'] = df['range_volume_correlation'] * df['amount_cluster_alignment']
    
    # Construct Final Alpha Factor
    df['core_breakout_signal'] = df['net_morning_breakout'] * df['net_afternoon_breakout']
    df['synchronization_weight'] = df['fractal_synchronization'] * df['volume_synchronization']
    df['momentum_decay_adjustment'] = df['opening_momentum_quality'] * df['momentum_decay_factor']
    
    df['final_factor'] = df['core_breakout_signal'] * df['synchronization_weight'] * df['momentum_decay_adjustment']
    
    return df['final_factor']
