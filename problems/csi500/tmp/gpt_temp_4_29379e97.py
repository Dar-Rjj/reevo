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
    
    # Calculate gaps
    data['gap'] = data['open'] - data['prev_close']
    data['gap_abs'] = np.abs(data['gap'])
    data['gap_direction'] = np.sign(data['gap'])
    
    # Calculate daily ranges
    data['daily_range'] = data['high'] - data['low']
    data['prev_daily_range'] = data['daily_range'].shift(1)
    data['range_5d'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    
    # Calculate 30-minute ranges (approximated as first 30 minutes of trading)
    # Using first hour data as proxy for 30-minute data
    data['high_30min'] = data['high'].rolling(window=2).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['low_30min'] = data['low'].rolling(window=2).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['range_30min'] = data['high_30min'] - data['low_30min']
    
    # Calculate volume features (using first hour as proxy for 30-minute volume)
    data['volume_first_30min'] = data['volume'].rolling(window=2).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['volume_concentration'] = data['volume_first_30min'] / data['volume']
    data['volume_median_10d'] = data['volume'].rolling(window=10).median()
    
    # Compression-Fracture Gap Analysis
    # Volatility-Scaled Gap Absorption
    data['gap_fill_efficiency'] = ((data['close'] - np.minimum(data['open'], data['prev_close'])) / 
                                  (data['gap_abs'] + 1e-8) * 
                                  data['range_30min'] / (data['high'].shift(2) - data['low'].shift(2) + 1e-8))
    
    data['gap_absorption_speed'] = (data['daily_range'] / (data['gap_abs'] + 1e-8) * 
                                   data['range_30min'] / (data['high'].shift(2) - data['low'].shift(2) + 1e-8))
    
    data['directional_absorption'] = (np.sign(data['close'] - data['open']) * 
                                     (data['gap_absorption_speed'] / (data['prev_daily_range'] + 1e-8)))
    
    # Compression-Fracture Gap Patterns
    data['gap_magnitude_compression'] = (data['gap_abs'] / (data['prev_high'] - data['prev_low'] + 1e-8) * 
                                        data['range_30min'] / (data['high'].shift(2) - data['low'].shift(2) + 1e-8))
    
    data['multi_day_gap_transition'] = (data['gap_abs'] / (data['gap_abs'].shift(1) + 1e-8) * 
                                       data['range_30min'] / (data['high'].shift(2) - data['low'].shift(2) + 1e-8))
    
    data['gap_direction_fracture'] = (data['gap_direction'] * 
                                     (np.abs(data['close'] - data['open']) / (data['gap_abs'] + 1e-8)))
    
    # Session Fracture Efficiency
    # Fracture Session Efficiency
    data['morning_fracture_efficiency'] = (data['high_30min'] - data['open']) / (data['range_30min'] + 1e-8)
    data['afternoon_fracture_efficiency'] = (data['close'] - data['low_30min']) / (data['range_30min'] + 1e-8)
    data['fracture_divergence'] = data['morning_fracture_efficiency'] - data['afternoon_fracture_efficiency']
    
    # Volume-Fracture Confirmation
    data['gap_volume_fracture'] = (data['volume'] / (data['prev_volume'] + 1e-8)) * data['fracture_divergence']
    data['volume_fracture_alignment'] = np.sign(data['fracture_divergence']) * data['volume_concentration']
    data['compression_volume_fracture'] = (data['volume'] / (data['volume_median_10d'] + 1e-8) * 
                                          data['range_30min'] / (data['high'].shift(2) - data['low'].shift(2) + 1e-8))
    
    # Fracture-Transition Momentum
    # Compression-Fracture Dynamics
    data['fracture_reversal'] = (np.sign(data['open'] - data['prev_close']) * 
                                np.sign(data['close'] - data['open']) * -1)
    
    data['compression_scaled_fracture'] = (data['fracture_reversal'] / (data['range_30min'] + 1e-8) / 
                                          (data['high'].shift(2) - data['low'].shift(2) + 1e-8))
    
    # Volatility-Compressed Session Dynamics
    data['range_compression'] = data['daily_range'] / (data['range_5d'] + 1e-8)
    data['compression_momentum'] = data['daily_range'] / (data['prev_daily_range'] + 1e-8)
    
    # Gap Absorption Fracture
    data['opening_gap_magnitude'] = data['gap'] / (data['prev_close'] + 1e-8)
    data['gap_fill_completeness'] = ((np.minimum(data['high'], data['prev_close']) - 
                                    np.maximum(data['low'], data['prev_close'])) / (data['gap_abs'] + 1e-8))
    
    # Volume-Weighted Fracture Momentum
    data['volume_timing_fracture'] = data['volume_concentration']
    data['price_fracture_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Calculate 5-day average true range
    data['true_range'] = np.maximum(
        np.maximum(data['high'] - data['low'], 
                  np.abs(data['high'] - data['prev_close'])),
        np.abs(data['low'] - data['prev_close'])
    )
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    
    data['volatility_scaled_fracture'] = data['price_fracture_5d'] / (data['atr_5d'] + 1e-8)
    data['volume_confirmed_fracture'] = data['price_fracture_5d'] * data['volume_timing_fracture']
    
    # Final factor integration
    # Combine key components with appropriate weights
    factor = (
        0.15 * data['gap_fill_efficiency'] +
        0.12 * data['directional_absorption'] +
        0.10 * data['gap_magnitude_compression'] +
        0.08 * data['fracture_divergence'] +
        0.10 * data['volume_fracture_alignment'] +
        0.12 * data['compression_scaled_fracture'] +
        0.08 * data['range_compression'] +
        0.10 * data['gap_fill_completeness'] +
        0.15 * data['volume_confirmed_fracture']
    )
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor.dropna()
