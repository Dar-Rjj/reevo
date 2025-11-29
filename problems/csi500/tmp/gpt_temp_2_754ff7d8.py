import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Session Momentum Elasticity and Efficiency Framework
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Morning session features (assuming first half of trading day)
    data['morning_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['morning_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['morning_close'] = data['close'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    
    # Afternoon session features (assuming second half of trading day)
    data['afternoon_open'] = data['open'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    data['afternoon_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    data['afternoon_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    
    # Morning Momentum Efficiency
    data['morning_efficiency'] = (data['morning_close'] - data['open']) / (data['morning_high'] - data['morning_low'] + 1e-8)
    
    # Afternoon Momentum Efficiency
    data['afternoon_efficiency'] = (data['close'] - data['afternoon_open']) / (data['afternoon_high'] - data['afternoon_low'] + 1e-8)
    
    # Cross-Session Efficiency Divergence
    data['efficiency_divergence'] = data['morning_efficiency'] - data['afternoon_efficiency']
    
    # Gap Momentum Efficiency
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_momentum_efficiency'] = data['opening_gap'] * data['morning_efficiency']
    
    # Volume-Enhanced Session Momentum
    data['morning_volume'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['afternoon_volume'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    
    data['volume_momentum_alignment'] = (
        (data['morning_efficiency'] * data['morning_volume']) - 
        (data['afternoon_efficiency'] * data['afternoon_volume'])
    ) / (data['morning_volume'] + data['afternoon_volume'] + 1e-8)
    
    # Price Level Momentum Elasticity Memory
    data['prev_close_efficiency'] = (data['close'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8)
    data['level_momentum_memory'] = data['morning_efficiency'] * data['prev_close_efficiency']
    
    # Volatility-Momentum Elasticity Interaction
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['prev_range'] = data['daily_range'].shift(1)
    
    data['volatility_momentum_alignment'] = (
        data['morning_efficiency'] * data['daily_range'] - 
        data['afternoon_efficiency'] * data['prev_range']
    )
    
    # Momentum Elasticity Reversal Detection
    data['momentum_exhaustion'] = (
        (data['morning_efficiency'].abs() - data['afternoon_efficiency'].abs()) * 
        np.sign(data['morning_efficiency'])
    )
    
    # Gap Momentum Reversal Signals
    data['gap_reversal_signal'] = data['opening_gap'] * data['efficiency_divergence']
    
    # Composite Momentum Elasticity Factor
    factors = [
        'efficiency_divergence',
        'gap_momentum_efficiency', 
        'volume_momentum_alignment',
        'level_momentum_memory',
        'volatility_momentum_alignment',
        'momentum_exhaustion',
        'gap_reversal_signal'
    ]
    
    # Normalize each factor component
    normalized_factors = []
    for factor in factors:
        if factor in data.columns:
            # Use rolling z-score for normalization (5-day window)
            mean = data[factor].rolling(window=5, min_periods=1).mean()
            std = data[factor].rolling(window=5, min_periods=1).std()
            normalized = (data[factor] - mean) / (std + 1e-8)
            normalized_factors.append(normalized)
    
    # Combine factors with equal weights
    if normalized_factors:
        composite_factor = sum(normalized_factors) / len(normalized_factors)
    else:
        composite_factor = pd.Series(index=data.index, data=0.0)
    
    # Final smoothing and cleaning
    final_factor = composite_factor.rolling(window=3, min_periods=1).mean()
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return final_factor
