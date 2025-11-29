import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Reversal Efficiency Spectrum Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Reversal Quality Assessment
    # Gap-Range Reversal Efficiency
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['daily_range'] = (data['high'] - data['low']) / data['open']
    data['gap_reversal_magnitude'] = np.abs(data['gap'])
    data['gap_range_ratio'] = data['gap_reversal_magnitude'] / (data['daily_range'] + 1e-8)
    
    # Gap reversal filling resistance strength
    data['gap_fill_ratio'] = np.where(
        data['gap'] > 0,
        (data['low'] - data['open']) / (data['open'] - data['close'].shift(1) + 1e-8),
        (data['high'] - data['open']) / (data['close'].shift(1) - data['open'] + 1e-8)
    )
    data['gap_reversal_strength'] = 1 - np.clip(np.abs(data['gap_fill_ratio']), 0, 1)
    
    # Gap reversal persistence
    data['close_vs_open'] = (data['close'] - data['open']) / data['open']
    data['gap_persistence'] = np.where(
        data['gap'] * data['close_vs_open'] < 0,
        np.abs(data['close_vs_open']) / (np.abs(data['gap']) + 1e-8),
        0
    )
    
    # Price Path Reversal Optimality
    data['high_low_range'] = data['high'] - data['low']
    data['open_close_range'] = np.abs(data['close'] - data['open'])
    data['reversal_efficiency'] = data['open_close_range'] / (data['high_low_range'] + 1e-8)
    
    # Early session reversal quality
    data['mid_session'] = (data['open'] + data['close']) / 2
    data['early_reversal_quality'] = np.where(
        data['gap'] * (data['mid_session'] - data['open']) < 0,
        np.abs(data['mid_session'] - data['open']) / (data['high_low_range'] + 1e-8),
        0
    )
    
    # Range Break Reversal Quality
    data['prev_range'] = (data['high'].shift(1) - data['low'].shift(1)) / data['close'].shift(2)
    data['range_compression'] = data['prev_range'].rolling(window=5).mean() / (data['prev_range'] + 1e-8)
    data['break_reversal_efficiency'] = data['reversal_efficiency'] * data['range_compression']
    
    # Volume-Price Reversal Dynamics
    # Volume Reversal Elasticity
    data['price_change'] = (data['close'] - data['open']) / data['open']
    data['volume_change'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['volume_elasticity'] = np.abs(data['price_change']) / (np.abs(data['volume_change']) + 1e-8)
    
    # Multi-timeframe volume reversal correlation
    data['volume_3d_ma'] = data['volume'].rolling(window=3).mean()
    data['volume_5d_ma'] = data['volume'].rolling(window=5).mean()
    data['volume_correlation'] = (
        data['volume'] / data['volume_3d_ma'] + 
        data['volume'] / data['volume_5d_ma']
    ) / 2
    
    # Amount-Based Reversal Impact
    data['reversal_per_amount'] = np.abs(data['price_change']) / (data['amount'] + 1e-8)
    data['amount_reversal_median'] = data['reversal_per_amount'].rolling(window=10).median()
    data['amount_reversal_deviation'] = data['reversal_per_amount'] / (data['amount_reversal_median'] + 1e-8)
    
    # Volume Reversal Cascade Confirmation
    data['volume_reversal_seq'] = (
        (data['volume_elasticity'].shift(1) > data['volume_elasticity'].shift(2)) & 
        (data['volume_elasticity'] > data['volume_elasticity'].shift(1))
    ).astype(float)
    data['volume_cascade_accel'] = data['volume_elasticity'].diff().rolling(window=3).mean()
    
    # Volatility-Adaptive Reversal Efficiency
    # Multi-Timeframe Reversal Consistency
    data['reversal_1d'] = data['reversal_efficiency']
    data['reversal_3d'] = data['reversal_efficiency'].rolling(window=3).mean()
    data['reversal_5d'] = data['reversal_efficiency'].rolling(window=5).mean()
    data['reversal_consistency'] = (
        data['reversal_1d'] / (data['reversal_3d'] + 1e-8) + 
        data['reversal_3d'] / (data['reversal_5d'] + 1e-8)
    ) / 2
    
    # Compression Breakout Reversal Signals
    data['volatility_5d'] = data['daily_range'].rolling(window=5).std()
    data['volatility_quantile'] = data['volatility_5d'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['compression_reversal'] = data['reversal_efficiency'] * (1 - data['volatility_quantile'])
    
    # Volatility-Specific Reversal Quality
    data['high_vol_reversal'] = np.where(
        data['volatility_quantile'] > 0.7,
        data['reversal_efficiency'] * data['volume_elasticity'],
        0
    )
    data['low_vol_reversal'] = np.where(
        data['volatility_quantile'] < 0.3,
        data['break_reversal_efficiency'] * data['range_compression'],
        0
    )
    
    # Stage-Based Reversal Weighting
    # Early Reversal Components
    early_reversal = (
        data['gap_range_ratio'] * 0.3 +
        data['gap_reversal_strength'] * 0.4 +
        data['volume_elasticity'] * 0.3
    )
    
    # Mid Reversal Components
    mid_reversal = (
        data['volume_reversal_seq'] * 0.4 +
        data['volume_cascade_accel'] * 0.3 +
        data['reversal_consistency'] * 0.3
    )
    
    # Late Reversal Components
    late_reversal = (
        (1 - data['gap_persistence']) * 0.4 +
        data['amount_reversal_deviation'] * 0.3 +
        data['compression_reversal'] * 0.3
    )
    
    # Multi-Dimensional Reversal Scoring
    gap_range_score = data['gap_range_ratio'] * data['gap_reversal_strength']
    volume_elasticity_score = data['volume_elasticity'] * data['volume_correlation']
    volatility_adaptive_score = (
        data['high_vol_reversal'] + data['low_vol_reversal'] + data['compression_reversal']
    )
    range_break_score = data['break_reversal_efficiency'] * data['range_compression']
    
    # Final Factor Synthesis
    factor = (
        early_reversal * 0.35 +
        mid_reversal * 0.35 +
        late_reversal * 0.3 +
        gap_range_score * 0.15 +
        volume_elasticity_score * 0.15 +
        volatility_adaptive_score * 0.15 +
        range_break_score * 0.15
    )
    
    # Apply volume-price alignment adjustments
    volume_alignment = data['volume_elasticity'] * data['amount_reversal_deviation']
    factor = factor * (1 + 0.2 * volume_alignment)
    
    # Return the factor series
    return factor
