import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Compression-Regime Analysis
    # Volatility-Adaptive Compression Detection
    data['daily_range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['daily_range'].rolling(window=5).mean()
    data['range_20d_median'] = data['daily_range'].rolling(window=20).median()
    
    # Range compression ratio
    data['compression_ratio'] = data['daily_range'] / data['range_5d_avg']
    
    # Volatility regime classification
    data['volatility_regime'] = np.where(data['daily_range'] > data['range_20d_median'], 1, 0)
    
    # Compression state (rank in 5-day window)
    data['compression_rank'] = data['daily_range'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    # Gap-Compression Dynamics
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = np.abs(data['open'] - data['prev_close']) / data['prev_close']
    data['gap_to_compression'] = data['opening_gap'] / (data['high'] - data['low'])
    data['regime_adjusted_gap'] = data['opening_gap'] * data['compression_rank']
    
    # Microstructure-Accelerated Momentum
    # Price-Microstructure Divergence
    data['return_2d'] = data['close'].pct_change(2)
    data['return_5d'] = data['close'].pct_change(5)
    data['price_acceleration'] = data['return_2d'] - data['return_5d']
    
    # Microstructure pressure (3-day change in daily range)
    data['microstructure_pressure'] = data['daily_range'].pct_change(3)
    data['acceleration_divergence'] = data['price_acceleration'] - data['microstructure_pressure']
    
    # Volume-Regime Alignment
    data['volume_3d_avg'] = data['volume'].rolling(window=3).mean()
    data['volume_momentum'] = (data['volume'] - data['volume_3d_avg']) / data['volume_3d_avg']
    data['volume_to_amplitude'] = data['volume'] / (data['high'] - data['low'])
    data['regime_adaptive_volume'] = data['volume_momentum'] * data['compression_rank']
    
    # Cross-Sectional Breakout Efficiency
    # Intraday Absorption Dynamics
    data['gap_closure_efficiency'] = (data['close'] - data['open']) / (data['open'] - data['prev_close'])
    data['intraday_absorption'] = (data['high'] - data['low']) / np.abs(data['open'] - data['prev_close'])
    
    # Cross-stock absorption (using sector correlation proxy - cross-sectional rank)
    data['gap_closure_rank'] = data['gap_closure_efficiency'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    # Compression-Breakout Confirmation
    data['closing_amplitude'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['breakout_strength'] = data['closing_amplitude'] * data['compression_rank']
    data['volume_confirmation'] = data['volume_to_amplitude'] * data['volume_momentum']
    data['microstructure_support'] = data['acceleration_divergence'] * data['breakout_strength']
    
    # Regime-Adaptive Signal Construction
    # Primary Momentum Component
    data['compression_adjusted_momentum'] = data['return_5d'] * data['compression_rank']
    data['gap_enhanced_momentum'] = data['compression_adjusted_momentum'] * data['opening_gap']
    data['microstructure_multiplier'] = data['gap_enhanced_momentum'] * data['acceleration_divergence']
    
    # Cross-Sectional Confirmation
    data['absorption_efficiency'] = data['gap_closure_efficiency'] * data['intraday_absorption']
    data['volume_regime_alignment'] = data['regime_adaptive_volume'] * data['volume_to_amplitude']
    
    # Cross-stock validation (using cross-sectional rank as proxy)
    data['absorption_rank'] = data['absorption_efficiency'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    # Adaptive Alpha Generation
    # Dynamic Component Integration
    # Volatility regime weighting
    regime_weight = np.where(data['volatility_regime'] == 1, 0.7, 1.3)
    
    # Microstructure acceleration scaling
    microstructure_scale = 1 + np.tanh(data['acceleration_divergence'])
    
    # Signal Combination and Enhancement
    # Combine momentum and confirmation components
    momentum_component = data['microstructure_multiplier'] * microstructure_scale
    confirmation_component = (data['absorption_efficiency'] + data['volume_regime_alignment']) * data['absorption_rank']
    
    # Apply directional bias using sign of closing amplitude
    directional_bias = np.sign(data['closing_amplitude'])
    
    # Scale by compression-to-expansion transition probability
    compression_transition = data['compression_ratio'].rolling(window=10).apply(
        lambda x: (x.iloc[-1] > x.iloc[-2]) * 1.0
    )
    
    # Final Cross-Sectional Factor
    factor = (momentum_component * 0.6 + confirmation_component * 0.4) * directional_bias * regime_weight
    factor = factor * (1 + compression_transition * 0.2)
    
    # Clean up and return
    result = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
