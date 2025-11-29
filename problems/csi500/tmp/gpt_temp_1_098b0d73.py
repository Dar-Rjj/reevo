import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Regime-Adaptive Momentum & Microstructure Alpha
    """
    data = df.copy()
    
    # Multi-Regime Detection & Classification
    # Volatility Regime Identification
    data['daily_range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['range_compression_ratio'] = data['daily_range'] / data['range_5d_avg']
    
    # Range compression state
    data['range_percentile'] = data['daily_range'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    data['volatility_regime'] = np.where(data['range_compression_ratio'] < 0.8, 'compressed', 
                                       np.where(data['range_compression_ratio'] > 1.2, 'expanded', 'normal'))
    
    # Microstructure Regime Assessment
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['amount_5d_avg'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['volume_pressure'] = data['volume'] / data['volume_5d_avg']
    data['amount_flow'] = data['amount'] / data['amount_5d_avg']
    
    data['microstructure_regime'] = np.where((data['volume_pressure'] > 1.1) & (data['amount_flow'] > 1.1), 'high_pressure',
                                           np.where((data['volume_pressure'] < 0.9) & (data['amount_flow'] < 0.9), 'low_pressure', 'normal'))
    
    # Combined regime state
    regime_mapping = {'compressed_high_pressure': 1, 'compressed_normal': 2, 'compressed_low_pressure': 3,
                     'normal_high_pressure': 4, 'normal_normal': 5, 'normal_low_pressure': 6,
                     'expanded_high_pressure': 7, 'expanded_normal': 8, 'expanded_low_pressure': 9}
    data['combined_regime'] = data['volatility_regime'] + '_' + data['microstructure_regime']
    data['regime_code'] = data['combined_regime'].map(regime_mapping).fillna(5)
    
    # Asymmetric Momentum Dynamics
    # Shadow Rejection Momentum
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['upper_shadow_dominance'] = data['upper_shadow'] / data['daily_range']
    data['lower_shadow_dominance'] = data['lower_shadow'] / data['daily_range']
    data['shadow_asymmetry'] = data['upper_shadow_dominance'] - data['lower_shadow_dominance']
    data['shadow_momentum'] = data['shadow_asymmetry'] * data['range_compression_ratio']
    
    # Multi-Timeframe Momentum Acceleration
    data['return_3d'] = data['close'].pct_change(periods=3)
    data['return_5d'] = data['close'].pct_change(periods=5)
    data['momentum_consistency'] = np.sign(data['return_3d']) * np.sign(data['return_5d'])
    
    # Handle division by zero
    denominator = np.abs(data['return_5d'])
    data['momentum_acceleration'] = np.where(denominator > 1e-8, 
                                           (data['return_3d'] - data['return_5d']) / denominator, 0)
    data['regime_momentum'] = data['momentum_acceleration'] * data['range_compression_ratio']
    
    # Gap Absorption Momentum
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    gap_denom = np.abs(data['open'] - data['prev_close'])
    data['absorption_efficiency'] = np.where(gap_denom > 1e-8, 
                                           (data['close'] - data['open']) / gap_denom, 0)
    data['gap_momentum'] = data['overnight_gap'] * data['absorption_efficiency'] * data['range_compression_ratio']
    
    # Volume-Confirmed Momentum
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    data['price_direction'] = np.sign(data['close'] - data['open'])
    data['volume_direction'] = np.sign(data['volume'] - data['prev_volume'])
    data['amount_direction'] = np.sign(data['amount'] - data['prev_amount'])
    
    data['volume_alignment'] = data['price_direction'] * data['volume_direction']
    data['amount_confirmation'] = data['price_direction'] * data['amount_direction']
    data['micro_momentum'] = data['volume_alignment'] * data['amount_confirmation']
    
    # Cross-Sectional Factor Construction
    # Regime-Weighted Component Integration
    data['compression_weight'] = np.where(data['volatility_regime'] == 'compressed', 1.5, 1.0)
    data['expansion_weight'] = np.where(data['volatility_regime'] == 'expanded', 1.5, 1.0)
    
    # Compression regime emphasis
    data['compression_component'] = (data['shadow_momentum'] + data['gap_momentum']) * data['compression_weight']
    
    # Expansion regime emphasis  
    data['expansion_component'] = (data['regime_momentum'] + data['micro_momentum']) * data['expansion_weight']
    
    # Dynamic regime weighting
    data['regime_weighted_signal'] = np.where(data['volatility_regime'] == 'compressed', data['compression_component'],
                                            np.where(data['volatility_regime'] == 'expanded', data['expansion_component'],
                                                   (data['compression_component'] + data['expansion_component']) / 2))
    
    # Microstructure Alignment
    data['price_micro_divergence'] = data['momentum_acceleration'] - data['volume_alignment']
    data['trading_pressure'] = data['amount_confirmation'] * data['volume_alignment']
    
    # Enhanced signal quality - positive when price leads microstructure
    data['enhanced_signal'] = np.where(data['price_micro_divergence'] > 0, 
                                     data['regime_weighted_signal'] * 1.2, 
                                     data['regime_weighted_signal'] * 0.8)
    
    # Final factor construction with regime persistence
    data['regime_persistence'] = data['regime_code'].rolling(window=3, min_periods=1).apply(
        lambda x: len(set(x)) == 1 if len(x) == 3 else False
    )
    
    data['final_factor'] = data['enhanced_signal'] * np.where(data['regime_persistence'], 1.1, 1.0)
    
    # Clean up and return
    factor_series = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factor_series
