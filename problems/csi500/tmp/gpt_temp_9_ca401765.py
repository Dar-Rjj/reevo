import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Regime Intraday Momentum Quality with Microstructural Anchoring
    """
    data = df.copy()
    
    # Multi-Timeframe Intraday Momentum Efficiency
    # Short-term momentum efficiency
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    data['momentum_efficiency'] = np.where(
        data['intraday_range'] > 0,
        data['intraday_move'] / data['intraday_range'] * data['volume'],
        0
    )
    
    # Medium-term momentum persistence (3-day average)
    data['momentum_persistence'] = data['momentum_efficiency'].rolling(window=3, min_periods=1).mean()
    
    # Long-term momentum quality (5-day consistency)
    data['momentum_consistency'] = data['momentum_efficiency'].rolling(window=5, min_periods=1).std()
    data['momentum_quality'] = data['momentum_persistence'] / (1 + data['momentum_consistency'])
    
    # Cross-timeframe momentum alignment
    short_rank = data['momentum_efficiency'].rolling(window=5, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    medium_rank = data['momentum_persistence'].rolling(window=5, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )
    data['momentum_alignment'] = 1 - np.abs(short_rank - medium_rank)
    
    # Volume-Volatility Elasticity with Regime Dependence
    # Intraday volatility efficiency
    data['volatility_efficiency'] = np.where(
        np.abs(data['intraday_move']) > 0,
        data['intraday_range'] / np.abs(data['intraday_move']),
        1
    )
    
    # Volume sensitivity to price range
    data['volume_sensitivity'] = np.where(
        data['intraday_range'] > 0,
        data['volume'] / data['intraday_range'],
        0
    )
    
    # Volatility regime classification
    data['volatility_regime'] = data['intraday_range'].div(data['close']).rolling(
        window=10, min_periods=1
    ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5)
    
    # Regime-specific elasticity patterns
    high_vol_regime = (data['volatility_regime'] > 0.7).astype(int)
    low_vol_regime = (data['volatility_regime'] < 0.3).astype(int)
    
    data['elasticity_high_vol'] = data['volume_sensitivity'] * high_vol_regime
    data['elasticity_low_vol'] = data['volume_sensitivity'] * low_vol_regime
    
    # Elasticity-momentum interaction
    data['elasticity_momentum'] = (
        data['momentum_efficiency'] * data['volume_sensitivity'] * 
        (1 + data['volatility_efficiency'])
    )
    
    # Microstructural Price Anchoring Dynamics
    # Opening price anchoring strength
    data['anchoring_strength'] = np.where(
        data['intraday_range'] > 0,
        np.abs(data['intraday_move']) / data['intraday_range'],
        0
    )
    
    # Intraday anchoring persistence (3-day average)
    data['anchoring_persistence'] = data['anchoring_strength'].rolling(window=3, min_periods=1).mean()
    
    # Volume concentration near key levels
    mid_price = (data['open'] + data['close']) / 2
    price_range = data['high'] - data['low']
    concentration_band = 0.1  # 10% of daily range
    
    # Simplified volume concentration (using close proximity to mid-price as proxy)
    data['volume_concentration'] = np.where(
        price_range > 0,
        np.abs(data['close'] - mid_price) / price_range,
        0
    )
    data['volume_concentration'] = 1 - data['volume_concentration']  # Higher when close to mid
    
    # Anchoring confidence
    data['anchoring_confidence'] = (
        data['anchoring_strength'] * 
        data['anchoring_persistence'] * 
        data['volume_concentration']
    )
    
    # Cross-Regime Transition Detection & Classification
    # Volume acceleration regime
    data['volume_acceleration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Momentum-volatility alignment
    data['momentum_vol_alignment'] = (
        data['momentum_efficiency'].rolling(window=5, min_periods=1).mean() *
        data['volatility_efficiency'].rolling(window=5, min_periods=1).mean()
    )
    
    # Regime classification
    range_expansion = (data['intraday_range'] > data['intraday_range'].rolling(window=5, min_periods=1).mean()).astype(int)
    volume_accel = (data['volume_acceleration'] > 1.2).astype(int)
    
    data['regime_shift_signal'] = range_expansion * volume_accel
    
    # Regime-adaptive weighting
    stable_regime = (
        (data['volatility_regime'].between(0.3, 0.7)) & 
        (data['volume_acceleration'].between(0.8, 1.2))
    ).astype(int)
    
    transition_regime = (data['regime_shift_signal'] == 1).astype(int)
    
    data['regime_weight'] = (
        0.6 * stable_regime + 
        0.3 * transition_regime + 
        0.1  # Base weight for other regimes
    )
    
    # Composite Factor Construction
    # Momentum quality score
    momentum_quality_score = (
        data['momentum_quality'] * 
        data['momentum_alignment'] * 
        data['momentum_efficiency']
    )
    
    # Volume-volatility confirmation
    volume_vol_confirmation = (
        data['elasticity_momentum'] * 
        data['volume_sensitivity'] * 
        (1 + data['volatility_efficiency'])
    )
    
    # Final cross-regime momentum factor
    composite_factor = (
        momentum_quality_score * 
        volume_vol_confirmation * 
        data['anchoring_confidence'] * 
        data['regime_weight']
    )
    
    # Normalize and return
    factor_series = composite_factor.rank(pct=True)
    return factor_series
