import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics with Regime-Based Anchoring
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Multi-Scale Fractal Dimension
    # High-low range complexity
    for window in [3, 5, 8]:
        # Price range fractal
        high_low_range = data['high'] - data['low']
        range_ma = high_low_range.rolling(window=window, min_periods=1).mean()
        range_std = high_low_range.rolling(window=window, min_periods=1).std()
        data[f'range_complexity_{window}'] = range_std / (range_ma + 1e-8)
        
        # Open-close tortuosity
        open_close_dev = (data['close'] - data['open']).abs()
        path_length = (data['high'] - data['low']).abs()
        data[f'tortuosity_{window}'] = open_close_dev.rolling(window=window, min_periods=1).sum() / \
                                      (path_length.rolling(window=window, min_periods=1).sum() + 1e-8)
        
        # Volume clustering patterns
        volume_ma = data['volume'].rolling(window=window, min_periods=1).mean()
        volume_std = data['volume'].rolling(window=window, min_periods=1).std()
        data[f'volume_clustering_{window}'] = volume_std / (volume_ma + 1e-8)
    
    # Combined fractal dimension
    data['fractal_dimension'] = (
        data['range_complexity_3'] + data['range_complexity_5'] + data['range_complexity_8'] +
        data['tortuosity_3'] + data['tortuosity_5'] + data['tortuosity_8'] +
        data['volume_clustering_3'] + data['volume_clustering_5'] + data['volume_clustering_8']
    ) / 9
    
    # 2. Identify Market Regime Transitions
    # Volatility regime using rolling percentiles
    volatility = data['high'] - data['low']
    vol_20d_ma = volatility.rolling(window=20, min_periods=1).mean()
    vol_20d_std = volatility.rolling(window=20, min_periods=1).std()
    vol_percentile = volatility.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    )
    
    # Price acceleration/deceleration
    returns = data['close'].pct_change()
    price_accel = returns.rolling(window=5, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    )
    
    # Volume expansion/contraction cycles
    volume_returns = data['volume'].pct_change()
    volume_cycle = volume_returns.rolling(window=10, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    )
    
    # Regime classification
    high_vol_regime = (vol_percentile > 1.0).astype(int)
    trending_regime = (price_accel.abs() > 1.0).astype(int)
    volume_expansion = (volume_cycle > 0.5).astype(int)
    
    # Combined regime indicator
    data['regime_score'] = high_vol_regime + trending_regime + volume_expansion
    
    # 3. Construct Regime-Adaptive Anchoring Signals
    # Regime-specific support/resistance levels
    for regime in [0, 1, 2, 3]:
        regime_mask = (data['regime_score'] == regime)
        if regime_mask.any():
            # Support level (recent lows within regime)
            regime_lows = data['low'][regime_mask].rolling(window=10, min_periods=1).min()
            data.loc[regime_mask, 'regime_support'] = regime_lows
            
            # Resistance level (recent highs within regime)
            regime_highs = data['high'][regime_mask].rolling(window=10, min_periods=1).max()
            data.loc[regime_mask, 'regime_resistance'] = regime_highs
    
    # Price momentum relative to current regime
    regime_price_range = data['regime_resistance'] - data['regime_support']
    price_position = (data['close'] - data['regime_support']) / (regime_price_range + 1e-8)
    
    # Volume confirmation across regime boundaries
    regime_volume_ma = data['volume'].rolling(window=20, min_periods=1).mean()
    volume_confirmation = data['volume'] / (regime_volume_ma + 1e-8)
    
    # Regime transition probability
    regime_changes = data['regime_score'].diff().abs()
    regime_stability = 1 - (regime_changes.rolling(window=10, min_periods=1).mean() / 3.0)
    
    # 4. Generate Fractal-Regime Alpha Factor
    # Combine fractal dimensions with regime probabilities
    fractal_regime_interaction = data['fractal_dimension'] * regime_stability
    
    # Regime-dependent signal weighting
    regime_weights = {
        0: 0.3,  # Low volatility, stable
        1: 0.6,  # Moderate regime changes
        2: 0.8,  # High activity
        3: 1.0   # Maximum regime complexity
    }
    
    regime_weight = data['regime_score'].map(regime_weights).fillna(0.5)
    
    # Multi-timeframe confirmation
    short_term_momentum = data['close'].pct_change(periods=3)
    medium_term_momentum = data['close'].pct_change(periods=8)
    momentum_confirmation = np.sign(short_term_momentum) * np.sign(medium_term_momentum)
    
    # Final factor construction
    factor = (
        fractal_regime_interaction * regime_weight * 
        price_position * volume_confirmation * 
        (1 + 0.2 * momentum_confirmation)
    )
    
    # Ensure no future data leakage
    factor = factor.shift(1).fillna(0)
    
    return factor
