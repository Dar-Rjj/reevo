import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['returns'] = data['close'].pct_change()
    data['high_low_ratio'] = data['high'] / data['low']
    data['close_open_ratio'] = data['close'] / data['open']
    
    # Volatility Regime Classification
    data['vol_3d'] = data['returns'].rolling(window=3).std()
    data['vol_10d'] = data['returns'].rolling(window=10).std()
    data['vol_ratio'] = data['vol_3d'] / data['vol_10d']
    data['high_vol_regime'] = (data['vol_ratio'] > 1.2).astype(int)
    data['low_vol_regime'] = (data['vol_ratio'] < 0.8).astype(int)
    
    # Regime-Specific Momentum
    # High volatility: Intraday return autocorrelation
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['return_autocorr'] = data['intraday_return'].rolling(window=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 5 else np.nan, raw=False
    )
    
    # Low volatility: Trend consistency (inverse variance of 3-period returns)
    data['return_var_3d'] = data['returns'].rolling(window=3).var()
    data['trend_consistency'] = 1 / (1 + data['return_var_3d'])
    
    # Combine regime-specific momentum
    data['regime_momentum'] = (
        data['high_vol_regime'] * data['return_autocorr'].fillna(0) +
        data['low_vol_regime'] * data['trend_consistency'].fillna(0)
    )
    
    # Volume-Price Divergence
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_deviation'] = (data['volume'] - data['volume_ma_5']) / data['volume_ma_5']
    data['price_deviation'] = (data['close'] - data['close'].rolling(window=5).mean()) / data['close'].rolling(window=5).mean()
    
    # High volume cluster price impact
    data['high_volume_cluster'] = (data['volume'] > data['volume_ma_5'] * 1.5).astype(int)
    data['cluster_price_impact'] = data['high_volume_cluster'] * data['returns'].abs()
    
    # Return sign vs volume deviation sign comparison
    data['return_sign'] = np.sign(data['returns'])
    data['volume_dev_sign'] = np.sign(data['volume_deviation'])
    data['volume_price_alignment'] = data['return_sign'] * data['volume_dev_sign']
    
    # Volume-price divergence magnitude
    data['volume_price_divergence'] = data['volume_price_alignment'] * data['cluster_price_impact']
    
    # Price-Level Behavior
    # Support/resistance proximity and compression
    data['rolling_high_5'] = data['high'].rolling(window=5).max()
    data['rolling_low_5'] = data['low'].rolling(window=5).min()
    data['price_range'] = data['rolling_high_5'] - data['rolling_low_5']
    data['price_position'] = (data['close'] - data['rolling_low_5']) / data['price_range']
    
    # Compression indicator (low range relative to average)
    data['avg_range_10'] = (data['high'] - data['low']).rolling(window=10).mean()
    data['range_compression'] = data['price_range'] / data['avg_range_10']
    
    # Round number proximity
    data['round_level_distance'] = np.abs(data['close'] - np.round(data['close'] / 10) * 10) / data['close']
    data['round_number_proximity'] = 1 / (1 + data['round_level_distance'])
    
    # Volume concentration near round numbers
    data['near_round_number'] = (data['round_level_distance'] < 0.005).astype(int)
    data['round_number_volume'] = data['near_round_number'] * data['volume']
    data['volume_concentration'] = data['round_number_volume'].rolling(window=5).sum() / data['volume'].rolling(window=5).sum()
    
    # Price level factor
    data['price_level_factor'] = (
        (1 - data['range_compression']) *  # Higher weight during compression
        data['price_position'] *  # Position within range
        data['round_number_proximity'] *  # Round number effect
        data['volume_concentration']  # Volume confirmation
    )
    
    # Adaptive Combination
    # Weight components by volatility regime
    high_vol_weight = data['high_vol_regime']
    low_vol_weight = data['low_vol_regime']
    
    # Combine factors with regime weighting
    regime_factor = (
        high_vol_weight * data['regime_momentum'] +
        low_vol_weight * data['price_level_factor']
    )
    
    # Multiply by volume-price divergence magnitude
    divergence_multiplier = 1 + data['volume_price_divergence'].abs()
    combined_factor = regime_factor * divergence_multiplier
    
    # Scale by current volume intensity
    volume_intensity = data['volume'] / data['volume_ma_5']
    final_factor = combined_factor * volume_intensity
    
    # Clean up and return
    result = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
