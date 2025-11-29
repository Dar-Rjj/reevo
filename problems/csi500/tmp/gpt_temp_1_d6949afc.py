import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday price levels (approximations)
    data['midday_high'] = data['high'].rolling(window=30, min_periods=1).apply(lambda x: x[:15].max() if len(x) >= 15 else np.nan)
    data['midday_low'] = data['low'].rolling(window=30, min_periods=1).apply(lambda x: x[:15].min() if len(x) >= 15 else np.nan)
    data['midday_price'] = (data['midday_high'] + data['midday_low']) / 2
    
    # Calculate morning session momentum
    data['morning_high_momentum'] = (data['midday_high'] - data['open']) / data['open']
    data['morning_low_momentum'] = (data['midday_low'] - data['open']) / data['open']
    data['morning_momentum'] = np.where(
        abs(data['morning_high_momentum']) > abs(data['morning_low_momentum']),
        data['morning_high_momentum'],
        data['morning_low_momentum']
    )
    
    # Calculate afternoon session momentum
    data['afternoon_momentum'] = (data['close'] - data['midday_price']) / data['midday_price']
    
    # Calculate intraday momentum divergence
    data['momentum_divergence'] = data['morning_momentum'] * data['afternoon_momentum'] * -1
    data['divergence_magnitude'] = abs(data['momentum_divergence'])
    data['divergence_signal'] = data['momentum_divergence'] * np.exp(data['divergence_magnitude'])
    
    # Calculate volume acceleration
    # Estimate morning volume (first half of trading hours)
    data['morning_volume'] = data['volume'].rolling(window=30, min_periods=1).apply(lambda x: x[:15].sum() if len(x) >= 15 else np.nan)
    data['volume_concentration'] = data['morning_volume'] / data['volume']
    
    # Calculate volume acceleration factor
    data['volume_concentration_roc'] = data['volume_concentration'].pct_change(periods=5)
    data['volume_acceleration'] = data['volume_concentration_roc'] * abs(data['morning_momentum'])
    
    # Combine divergence with volume acceleration
    data['divergence_volume_composite'] = data['divergence_signal'] * data['volume_acceleration']
    
    # Volatility clustering and regime detection
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['volatility_clustering'] = data['intraday_range'].rolling(window=20).var()
    data['volatility_regime'] = data['volatility_clustering'].rolling(window=10).mean()
    
    # Generate regime adjustment factor
    volatility_median = data['volatility_regime'].median()
    data['regime_adjustment'] = np.where(
        data['volatility_regime'] > volatility_median,
        0.7,  # Dampen in high volatility
        1.3   # Amplify in low volatility
    )
    
    # Final factor construction with time-decay weighting
    data['raw_factor'] = data['divergence_volume_composite'] * data['regime_adjustment']
    
    # Apply exponential time decay to recent signals (5-day window)
    decay_weights = np.exp(-np.arange(5) / 2.5)  # Exponential decay
    decay_weights = decay_weights / decay_weights.sum()  # Normalize
    
    data['final_factor'] = data['raw_factor'].rolling(window=5, min_periods=1).apply(
        lambda x: np.dot(x, decay_weights[:len(x)]) if len(x) > 0 else np.nan
    )
    
    # Generate directional output based on session alignment
    data['session_alignment'] = np.sign(data['morning_momentum']) * np.sign(data['afternoon_momentum'])
    data['factor_output'] = np.where(
        data['session_alignment'] > 0,
        data['final_factor'],
        -data['final_factor']
    )
    
    # Handle NaN values and return the factor
    factor = data['factor_output'].fillna(0)
    return factor
