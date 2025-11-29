import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price Compression Measurement
    # Calculate Intraday Range Compression
    data['current_range'] = data['high'] - data['low']
    data['avg_range_5d'] = data['current_range'].rolling(window=5, min_periods=3).mean()
    data['range_compression_ratio'] = data['current_range'] / data['avg_range_5d']
    
    # Compute Opening Compression Signal
    data['prev_low'] = data['low'].shift(1)
    data['opening_range'] = data['open'] - data['prev_low']
    data['avg_opening_range_5d'] = data['opening_range'].rolling(window=5, min_periods=3).mean()
    data['opening_compression'] = data['opening_range'] / data['avg_opening_range_5d']
    
    # Form Compression Composite
    data['compression_score'] = data['range_compression_ratio'] * data['opening_compression']
    
    # Dynamic Volume Confirmation System
    # Volume Distribution Analysis
    data['volume_percentile'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_change'] = data['volume'] / data['prev_volume']
    data['avg_volume_change_5d'] = data['volume_change'].rolling(window=5, min_periods=3).mean()
    data['volume_acceleration'] = data['volume_change'] / data['avg_volume_change_5d']
    
    # Amount-Volume Divergence Detection
    data['amount_per_share'] = data['amount'] / data['volume']
    data['avg_amount_per_share_10d'] = data['amount_per_share'].rolling(window=10, min_periods=5).mean()
    data['amount_premium'] = data['amount_per_share'] / data['avg_amount_per_share_10d']
    
    # Volume Confirmation Signal
    data['volume_confidence'] = data['volume_percentile'] * data['volume_acceleration'] * data['amount_premium']
    
    # Breakout Direction and Strength Assessment
    # High-Side Breakout Potential
    data['high_10d_max'] = data['high'].rolling(window=10, min_periods=5).max()
    data['low_10d_min'] = data['low'].rolling(window=10, min_periods=5).min()
    data['high_proximity'] = (data['high_10d_max'] - data['high']) / (data['high_10d_max'] - data['low_10d_min'] + 1e-8)
    
    data['close_to_high_ratio'] = (data['close'] - data['open']) / (data['high'] - data['open'] + 1e-8)
    data['high_breakout_momentum'] = data['close_to_high_ratio'] * data['high_proximity']
    
    # Low-Side Breakout Potential
    data['low_proximity'] = (data['low'] - data['low_10d_min']) / (data['high_10d_max'] - data['low_10d_min'] + 1e-8)
    
    data['close_to_low_ratio'] = (data['open'] - data['close']) / (data['open'] - data['low'] + 1e-8)
    data['low_breakout_momentum'] = data['close_to_low_ratio'] * data['low_proximity']
    
    # Net Breakout Direction
    data['breakout_bias'] = data['high_breakout_momentum'] - data['low_breakout_momentum']
    
    # Intraday Efficiency and Persistence
    # Price Efficiency Measurement
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['efficiency_ratio'] = np.abs(data['close'] - data['open']) / (data['true_range'] + 1e-8)
    
    # Compute Trend Consistency
    data['intraday_trend'] = np.sign(data['close'] - data['open'])
    data['prev_open'] = data['open'].shift(1)
    data['prev_intraday_trend'] = np.sign(data['prev_close'] - data['prev_open'])
    data['trend_persistence'] = (data['intraday_trend'] == data['prev_intraday_trend']).astype(float)
    
    # Efficiency Adjustment Factor
    data['efficiency_score'] = data['efficiency_ratio'] * data['trend_persistence']
    
    # Multi-Factor Integration Logic
    # Compression-Volume Alignment
    compression_volume_multiplier = np.ones(len(data))
    high_comp_high_vol_mask = (data['compression_score'] < 0.8) & (data['volume_confidence'] > 1.2)
    low_comp_low_vol_mask = (data['compression_score'] > 1.2) & (data['volume_confidence'] < 0.8)
    
    compression_volume_multiplier[high_comp_high_vol_mask] = 1.4
    compression_volume_multiplier[low_comp_low_vol_mask] = 0.6
    
    # Breakout-Efficiency Enhancement
    breakout_efficiency_multiplier = np.ones(len(data))
    strong_breakout_high_eff_mask = (np.abs(data['breakout_bias']) > 0.3) & (data['efficiency_score'] > 0.7)
    weak_breakout_low_eff_mask = (np.abs(data['breakout_bias']) < 0.1) & (data['efficiency_score'] < 0.3)
    
    breakout_efficiency_multiplier[strong_breakout_high_eff_mask] = 1.3
    breakout_efficiency_multiplier[weak_breakout_low_eff_mask] = 0.7
    
    # Final Composite Factor Construction
    data['core_factor'] = data['compression_score'] * data['breakout_bias']
    data['volume_weighted_core'] = data['core_factor'] * data['volume_confidence']
    
    # Apply all multipliers and adjustments
    data['final_adjustment'] = (
        data['volume_weighted_core'] * 
        compression_volume_multiplier * 
        breakout_efficiency_multiplier * 
        data['efficiency_score']
    )
    
    # Return the final factor series
    return data['final_adjustment']
