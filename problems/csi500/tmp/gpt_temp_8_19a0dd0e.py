import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['daily_range'] = (data['high'] - data['low']) / data['open']
    data['close_to_open'] = (data['close'] - data['open']) / data['open']
    
    # Gap Reversal Strength Analysis
    data['gap_direction'] = np.sign(data['gap'])
    data['gap_magnitude'] = np.abs(data['gap'])
    
    # 3-day gap reversal rate
    data['gap_reversal_3d'] = 0
    for i in range(2, len(data)):
        if i >= 2:
            current_gap = data['gap_direction'].iloc[i]
            prev_gaps = data['gap_direction'].iloc[i-2:i]
            reversals = sum(prev_gaps * current_gap < 0)
            data.iloc[i, data.columns.get_loc('gap_reversal_3d')] = reversals / 2.0
    
    # Gap-Volume Integration
    data['gap_volume_confirmation'] = data['gap_magnitude'] * data['volume'] / data['volume'].rolling(20).mean()
    
    # Gap Rejection Signals
    data['rejection_ratio'] = np.abs(data['gap']) / (data['daily_range'] + 1e-8)
    data['gap_rejection_signal'] = (data['rejection_ratio'] > 0.5) & (data['gap_direction'] * data['close_to_open'] < 0)
    
    # Intraday Price Path Efficiency
    data['min_distance'] = np.abs(data['close'] - data['open']) / data['open']
    data['actual_distance'] = ((data['high'] - data['low']) + np.abs(data['gap'] * data['open'])) / data['open']
    data['efficiency_ratio'] = data['min_distance'] / (data['actual_distance'] + 1e-8)
    
    # Efficiency-Volume Integration
    data['volume_weighted_efficiency'] = data['efficiency_ratio'] * data['volume'] / data['volume'].rolling(20).mean()
    
    # Price-Volume Rejection Asymmetry
    data['opening_rejection'] = np.where(
        data['open'] > data['prev_close'],
        (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8),
        (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    )
    
    data['closing_absorption'] = np.where(
        data['close'] > data['open'],
        (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8),
        (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    )
    
    data['rejection_asymmetry'] = data['opening_rejection'] - data['closing_absorption']
    data['volume_rejection_bias'] = data['rejection_asymmetry'] * (data['volume'] / data['volume'].rolling(10).mean() - 1)
    
    # Range Persistence Momentum
    data['range_persistence'] = data['daily_range'].rolling(5).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1].mean()) * 1.0 if len(x) == 5 else 0
    )
    
    data['range_volume_confirmation'] = data['range_persistence'] * (data['volume'] / data['volume'].rolling(10).mean())
    
    # Range breakout detection
    data['range_compression'] = data['daily_range'].rolling(5).std() / (data['daily_range'].rolling(5).mean() + 1e-8)
    data['breakout_signal'] = (data['range_compression'] < 0.3) & (data['volume'] > data['volume'].rolling(10).mean() * 1.2)
    
    # Composite Gap Rejection Efficiency Factor
    # Core Gap Rejection Component
    gap_reversal_component = data['gap_reversal_3d'] * data['efficiency_ratio'] * data['range_persistence']
    
    # Volume Integration Enhancement
    volume_component = data['gap_volume_confirmation'] * data['volume_weighted_efficiency'] * data['volume_rejection_bias']
    
    # Range persistence confirmation
    range_component = data['range_volume_confirmation'] * (1 + data['breakout_signal'] * 0.5)
    
    # Final composite factor
    composite_factor = (
        gap_reversal_component * 
        np.sign(data['gap_direction']) * 
        (1 + volume_component) * 
        range_component
    )
    
    # Handle NaN values
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan)
    
    return composite_factor
