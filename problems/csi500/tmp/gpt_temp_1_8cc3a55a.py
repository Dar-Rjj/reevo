import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Embedded Opening Momentum Efficiency (VOOME) factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Opening Momentum in Volatility Spectrum
    # Core Opening Efficiency Components
    data['opening_momentum_intensity'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['gap_absorption_efficiency'] = np.abs(data['open'] - data['prev_close']) / (data['open'] - data['low'] + 1e-8)
    data['opening_range_utilization'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Volatility-Embedded Opening Significance
    data['opening_volatility_ratio'] = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    
    # Previous day's opening momentum for persistence calculation
    prev_opening_momentum = (data['open'].shift(1) - data['prev_low']) / (data['prev_high'] - data['prev_low'] + 1e-8)
    data['momentum_persistence'] = data['opening_momentum_intensity'] - prev_opening_momentum
    
    # Volatility-adaptive momentum impact
    volatility_context = data['opening_volatility_ratio'].rolling(window=5, min_periods=3).mean()
    data['volatility_adaptive_momentum'] = data['opening_momentum_intensity'] * (1 + volatility_context)
    
    # Intraday Absorption & Momentum Dynamics
    # Absorption Strength Analysis
    data['upper_absorption'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['lower_absorption'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['net_absorption_bias'] = data['upper_absorption'] - data['lower_absorption']
    
    # Momentum Efficiency Measurement
    data['volume_momentum_concentration'] = data['volume'] / (np.abs(data['open'] - data['prev_close']) + 1e-8)
    data['opening_impact_assessment'] = np.abs(data['open'] - data['prev_close']) / (data['volume'] + 1e-8)
    
    # Opening session momentum strength
    data['opening_momentum_strength'] = np.abs(data['open'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8)
    
    # Midday absorption sustainability
    data['midday_absorption'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Momentum regime detection
    momentum_regime = np.where(
        (data['opening_momentum_strength'] > data['opening_momentum_strength'].rolling(window=10).quantile(0.7)) & 
        (data['net_absorption_bias'].abs() < 0.3), 
        1,  # High momentum
        np.where(
            (data['opening_momentum_strength'] < data['opening_momentum_strength'].rolling(window=10).quantile(0.3)) & 
            (data['net_absorption_bias'].abs() > 0.6), 
            -1,  # Low momentum
            0   # Normal momentum
        )
    )
    data['momentum_regime'] = momentum_regime
    
    # Liquidity & Momentum Confirmation
    # Embedded Liquidity Dynamics
    data['opening_volume_expansion'] = data['volume'] / (data['prev_volume'] + 1e-8)
    data['volume_momentum_regime'] = data['volume'] / (np.abs(data['open'] - data['prev_close']) + 1e-8)
    data['liquidity_momentum'] = data['opening_volume_expansion'] * data['volume_momentum_regime']
    
    # Momentum Continuation Assessment
    data['momentum_continuation_pct'] = np.abs(data['close'] - data['open']) / (np.abs(data['open'] - data['prev_close']) + 1e-8)
    data['continuation_speed'] = data['momentum_continuation_pct']
    
    # Continuation direction analysis
    opening_direction = np.sign(data['open'] - data['prev_close'])
    close_direction = np.sign(data['close'] - data['open'])
    data['momentum_continuation'] = (opening_direction == close_direction).astype(int)
    data['momentum_reversal'] = (opening_direction != close_direction).astype(int)
    
    # Range Utilization & Momentum Patterns
    # Volatility-Momentum Relationship
    data['opening_range_ratio'] = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    
    # Momentum utilization within daily range
    data['momentum_range_utilization'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Multi-Timeframe Momentum Context
    data['opening_momentum_persistence_3d'] = data['opening_momentum_intensity'].rolling(window=3).mean()
    data['volatility_regime_consistency'] = data['opening_volatility_ratio'].rolling(window=5).std()
    
    # Momentum degradation detection
    momentum_trend = data['opening_momentum_intensity'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    data['momentum_degradation'] = np.where(
        (momentum_trend < 0) & (data['high'] - data['low'] < data['high'].rolling(window=10).mean() - data['low'].rolling(window=10).mean()),
        -1, 0
    )
    
    # Adaptive Composite Signal Construction
    # Primary Opening Momentum Component
    primary_momentum = (
        data['volatility_adaptive_momentum'] * 
        (1 + data['opening_range_utilization']) *
        np.where(data['momentum_regime'] == 1, 1.2, np.where(data['momentum_regime'] == -1, 0.8, 1.0))
    )
    
    # Absorption-Momentum Component
    absorption_momentum = (
        data['net_absorption_bias'] * 
        (data['opening_momentum_intensity'] - data['opening_momentum_intensity'].rolling(window=10).mean()) *
        data['volume_momentum_concentration'] / (data['volume_momentum_concentration'].rolling(window=20).std() + 1e-8)
    )
    
    # Confirmation Framework
    confirmation = (
        data['liquidity_momentum'] / (data['liquidity_momentum'].rolling(window=20).std() + 1e-8) *
        data['momentum_continuation_pct'] *
        data['momentum_range_utilization']
    )
    
    # Final Composite Signal
    final_signal = (
        primary_momentum * 
        absorption_momentum * 
        confirmation *
        (1 + 0.1 * data['momentum_continuation']) *
        (1 - 0.15 * data['momentum_reversal']) *
        (1 - 0.1 * np.abs(data['momentum_degradation']))
    )
    
    # Normalize the final signal
    normalized_signal = (final_signal - final_signal.rolling(window=50).mean()) / (final_signal.rolling(window=50).std() + 1e-8)
    
    return normalized_signal
