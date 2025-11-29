import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Rejection Framework
    data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['high']
    data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['low']
    
    data['upper_rejection_3d'] = data['upper_shadow'].rolling(window=3, min_periods=1).mean()
    data['lower_support_3d'] = data['lower_shadow'].rolling(window=3, min_periods=1).mean()
    
    data['price_body'] = np.abs(data['close'] - data['open'])
    data['total_range'] = data['high'] - data['low']
    data['body_ratio'] = data['price_body'] / data['total_range']
    data['body_ratio'] = data['body_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Gap Momentum Analysis
    data['prev_close'] = data['close'].shift(1)
    data['gap_pct'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_magnitude'] = np.abs(data['gap_pct'])
    
    data['return_5d'] = data['close'].pct_change(periods=5)
    data['gap_momentum_divergence'] = data['gap_pct'] - data['return_5d']
    
    data['gap_direction'] = np.sign(data['gap_pct'])
    data['gap_direction_consistency'] = data['gap_direction'].rolling(window=3, min_periods=1).sum() / 3
    data['gap_3d_avg'] = data['gap_pct'].rolling(window=3, min_periods=1).mean()
    data['gap_momentum'] = data['gap_pct'] / data['gap_3d_avg']
    data['gap_momentum'] = data['gap_momentum'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volatility Adjustment Framework
    data['daily_range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    
    data['vol_adj_upper_rejection'] = (data['upper_rejection_3d'] / data['daily_range']) * data['range_5d_avg']
    data['vol_adj_lower_support'] = (data['lower_support_3d'] / data['daily_range']) * data['range_5d_avg']
    data['vol_adj_lower_support'] = data['vol_adj_lower_support'] * data['body_ratio']
    
    # Volume Confirmation System
    data['volume_intensity'] = data['volume'] / (data['high'] - data['low'])
    data['volume_intensity'] = data['volume_intensity'].replace([np.inf, -np.inf], 0).fillna(0)
    data['volume_concentration_3d'] = data['volume_intensity'].rolling(window=3, min_periods=1).mean()
    
    data['gap_volume_ratio'] = data['volume_intensity'] / (data['gap_magnitude'] + 1e-8)
    data['volume_gap_coherence'] = np.sign(data['gap_pct']) * data['volume_intensity']
    
    data['volume_trend'] = data['volume'].rolling(window=3, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 2 else 0
    )
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['volume_acceleration'] = data['volume'] / (data['volume_3d_avg'] + 1e-8)
    
    # Multi-timeframe Integration
    data['high_low_range'] = data['high'] - data['low']
    data['range_persistence_2d'] = data['high_low_range'].rolling(window=2, min_periods=1).std()
    
    data['gap_pattern_alignment'] = data['gap_pct'].rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x[i]) == np.sign(x[i-1])]) / max(1, len(x)-1)
    )
    
    data['range_expansion'] = data['daily_range'] / data['range_5d_avg']
    data['volatility_persistence'] = data['daily_range'].rolling(window=5, min_periods=1).std() / data['range_5d_avg']
    
    # Signal Synthesis and Amplification
    data['vol_rejection_amp'] = (
        data['vol_adj_upper_rejection'] * 
        data['gap_momentum'] * 
        data['gap_volume_ratio'] * 
        data['body_ratio']
    )
    
    data['vol_support_enhance'] = (
        data['vol_adj_lower_support'] * 
        data['gap_direction_consistency'] * 
        data['volume_acceleration'] * 
        data['gap_pattern_alignment']
    )
    
    data['gap_momentum_bias'] = (
        data['gap_momentum_divergence'] * 
        data['volume_gap_coherence'] * 
        (1 - data['volatility_persistence'])
    )
    
    data['volume_confirmation'] = (
        data['volume_concentration_3d'] * 
        data['volume_trend'] * 
        data['range_persistence_2d']
    )
    
    # Final Alpha Construction
    data['final_alpha'] = (
        (data['vol_rejection_amp'] - data['vol_support_enhance']) * 
        data['gap_momentum_bias'] * 
        data['volume_confirmation']
    )
    
    # Clean up and return
    result = data['final_alpha'].replace([np.inf, -np.inf], 0).fillna(0)
    return result
