import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Range Momentum Quality
    data['daily_range'] = (data['high'] - data['low']) / data['prev_close']
    data['prev_range'] = (data['prev_high'] - data['prev_low']) / data['prev_close'].shift(1)
    data['range_persistence'] = data['daily_range'] / (data['prev_range'] + 1e-8)
    
    # Range-Price Alignment
    data['price_change'] = (data['close'] - data['prev_close']) / data['prev_close']
    data['range_expansion_up'] = ((data['daily_range'] > data['prev_range']) & 
                                 (data['price_change'] > 0)).astype(float)
    data['range_contraction_stable'] = ((data['daily_range'] < data['prev_range']) & 
                                       (abs(data['price_change']) < 0.01)).astype(float)
    
    # Gap Momentum Efficiency
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_to_range_ratio'] = abs(data['opening_gap']) / (data['daily_range'] + 1e-8)
    
    # Gap Filling Quality
    data['intraday_high_capture'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['intraday_low_capture'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['close_gap_position'] = (data['close'] - data['prev_close']) / (data['open'] - data['prev_close'] + 1e-8)
    
    # Volume-Momentum Synchronization
    data['early_volume_intensity'] = data['volume'] / (data['prev_volume'] + 1e-8)
    data['late_volume_momentum'] = data['volume'].rolling(window=3, min_periods=1).mean() / data['prev_volume']
    
    # Amount Efficiency
    data['price_per_amount'] = data['price_change'] / (data['amount'] + 1e-8)
    data['amount_momentum'] = data['amount'] / data['prev_amount']
    
    # Multi-day Efficiency Consistency
    data['range_momentum_3d'] = data['daily_range'].rolling(window=3, min_periods=1).mean()
    data['volume_confirmation'] = (data['volume'] > data['volume'].rolling(window=5, min_periods=1).mean()).astype(float)
    
    # Calculate composite factors
    data['range_momentum_quality'] = (
        data['range_persistence'] * 0.3 +
        data['range_expansion_up'] * 0.3 +
        data['range_contraction_stable'] * 0.4
    )
    
    data['gap_momentum_efficiency'] = (
        data['opening_gap'] * 0.4 +
        (1 - data['gap_to_range_ratio']) * 0.3 +
        data['intraday_high_capture'] * 0.15 +
        data['intraday_low_capture'] * 0.15
    )
    
    data['volume_synchronization'] = (
        np.log1p(data['early_volume_intensity']) * 0.4 +
        np.log1p(data['late_volume_momentum']) * 0.3 +
        data['price_per_amount'] * 0.15 +
        np.log1p(data['amount_momentum']) * 0.15
    )
    
    # Final composite factor
    data['momentum_efficiency_factor'] = (
        data['range_momentum_quality'] * 0.35 +
        data['gap_momentum_efficiency'] * 0.35 +
        data['volume_synchronization'] * 0.3
    )
    
    # Cross-sectional ranking (z-score normalization)
    factor = data.groupby(data.index)['momentum_efficiency_factor'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    return factor
