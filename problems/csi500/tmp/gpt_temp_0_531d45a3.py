import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Regime-Adaptive Momentum Efficiency factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Identify Volatility Regime
    # Compute Short-Term Volatility
    short_vol = data['returns'].rolling(window=3, min_periods=2).std()
    short_vol_smooth = short_vol.ewm(span=5, min_periods=3).mean()
    
    # Compute Medium-Term Volatility
    medium_vol = data['returns'].rolling(window=10, min_periods=5).std()
    medium_vol_smooth = medium_vol.ewm(span=10, min_periods=5).mean()
    
    # Determine Regime Classification
    vol_ratio = short_vol_smooth / medium_vol_smooth
    high_vol_regime = (vol_ratio > 1.2).astype(int)
    low_vol_regime = (vol_ratio < 0.8).astype(int)
    transition_regime = ((vol_ratio >= 0.8) & (vol_ratio <= 1.2)).astype(int)
    
    # 2. Calculate Regime-Specific Momentum Component
    
    # For intraday calculations, we'll use OHLC data to approximate intraday behavior
    # Calculate intraday range and position
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['intraday_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # High Volatility Regime Momentum
    # Intraday Return Persistence approximation
    data['hourly_returns_approx'] = data['intraday_range'] * data['intraday_position']
    autocorr_persistence = data['hourly_returns_approx'].rolling(window=3, min_periods=2).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    ).abs()
    
    # Volatility-Adjusted Reversal
    rolling_3day_low = data['low'].rolling(window=3, min_periods=2).min()
    rolling_3day_high = data['high'].rolling(window=3, min_periods=2).max()
    reversal_component = (data['close'] - rolling_3day_low) / (rolling_3day_high - rolling_3day_low).replace(0, np.nan)
    
    # True Range Volatility
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    true_range_avg = true_range.rolling(window=5, min_periods=3).mean()
    
    volatility_adjusted_reversal = reversal_component / true_range_avg.replace(0, np.nan)
    
    high_vol_momentum = autocorr_persistence * volatility_adjusted_reversal
    
    # Low Volatility Regime Momentum
    # Trend Consistency approximation
    data['intraday_trend'] = (data['close'] - data['open']) / data['open']
    trend_variance = data['intraday_trend'].rolling(window=3, min_periods=2).var()
    trend_consistency = 1 / (trend_variance.replace(0, np.nan))
    
    # Intraday Momentum Quality
    intraday_return = (data['close'] - data['open']) / data['open']
    price_acceleration = intraday_return.diff()
    acceleration_sign_consistency = price_acceleration.rolling(window=3, min_periods=2).apply(
        lambda x: abs(sum(np.sign(x))) / len(x) if len(x) > 0 else np.nan, raw=False
    )
    
    low_vol_momentum = trend_consistency * acceleration_sign_consistency
    
    # Combine Regime Momentum
    regime_momentum = (
        high_vol_regime * high_vol_momentum + 
        low_vol_regime * low_vol_momentum + 
        transition_regime * (high_vol_momentum + low_vol_momentum) / 2
    )
    regime_momentum_smooth = regime_momentum.rolling(window=5, min_periods=3).mean()
    
    # 3. Calculate Volume-Price Divergence Filter
    
    # Volume Clustering Effect
    avg_volume_5day = data['volume'].rolling(window=5, min_periods=3).mean()
    high_volume_flag = (data['volume'] > 1.5 * avg_volume_5day).astype(int)
    
    # Price Impact per High Volume Cluster
    price_change_high_vol = (data['close'] - data['open']) * high_volume_flag
    volume_impact = price_change_high_vol / data['volume'].replace(0, np.nan)
    volume_clustering_effect = volume_impact.rolling(window=3, min_periods=2).sum()
    
    # Return-Volume Divergence
    return_sign = np.sign(data['close'] - data['open'])
    volume_deviation_sign = np.sign(data['volume'] - avg_volume_5day)
    divergence_magnitude = abs(return_sign - volume_deviation_sign) / 2
    
    # Generate Volume Divergence Score
    divergence_score = volume_clustering_effect * divergence_magnitude
    divergence_score_cubic = np.sign(divergence_score) * (abs(divergence_score) ** (1/3))
    
    # Scale by daily volume percentile
    volume_percentile = data['volume'].rolling(window=20, min_periods=10).rank(pct=True)
    volume_divergence = divergence_score_cubic * volume_percentile
    
    # 4. Calculate Price-Level Dependent Behavior
    
    # Support/Resistance Proximity
    recent_high = data['high'].rolling(window=10, min_periods=5).max()
    recent_low = data['low'].rolling(window=10, min_periods=5).min()
    
    dist_to_high = (recent_high - data['close']) / data['close']
    dist_to_low = (data['close'] - recent_low) / data['close']
    support_resistance_proximity = np.minimum(dist_to_high, dist_to_low)
    
    # Price Compression Near Extremes
    price_range = (data['high'] - data['low']) / data['close']
    avg_range = price_range.rolling(window=10, min_periods=5).mean()
    compression_ratio = price_range / avg_range.replace(0, np.nan)
    
    support_resistance_effect = support_resistance_proximity * compression_ratio
    
    # Round Number Effect
    def round_number_proximity(price):
        round_levels = [0.50, 1.00, 5.00, 10.00, 50.00, 100.00]
        distances = [abs(price - round(price / level) * level) / price for level in round_levels]
        return min(distances) if distances else 0
    
    round_proximity = data['close'].apply(round_number_proximity)
    
    # Volume Concentration at Round Numbers
    round_number_cross = (data['close'].apply(lambda x: any(abs(x - round(x / level) * level) < 0.01 * x for level in [0.50, 1.00, 5.00])).astype(int))
    volume_at_round = data['volume'] * round_number_cross
    avg_volume_non_round = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_concentration = volume_at_round / avg_volume_non_round.replace(0, np.nan)
    
    round_number_effect = round_proximity * volume_concentration
    
    # Generate Price-Level Signal
    price_level_signal = support_resistance_effect * round_number_effect
    price_level_signal_smooth = price_level_signal.ewm(span=3, min_periods=2).mean()
    
    # 5. Combine All Components Adaptively
    
    # Calculate Component Weights by Regime
    momentum_weight = high_vol_regime * 0.6 + low_vol_regime * 0.3 + transition_regime * 0.45
    volume_weight = high_vol_regime * 0.2 + low_vol_regime * 0.5 + transition_regime * 0.35
    price_weight = high_vol_regime * 0.2 + low_vol_regime * 0.2 + transition_regime * 0.2
    
    # Generate Composite Signal
    composite_signal = (
        momentum_weight * regime_momentum_smooth +
        volume_weight * volume_divergence +
        price_weight * price_level_signal_smooth
    )
    
    # Apply regime-specific scaling
    regime_scaling = high_vol_regime * 0.8 + low_vol_regime * 1.2 + transition_regime * 1.0
    scaled_signal = composite_signal * regime_scaling
    
    # Final Signal Processing
    # Apply Volume-Price Divergence Weighting
    divergence_magnitude_weight = 1 + abs(volume_divergence)
    signal_with_divergence = scaled_signal * divergence_magnitude_weight
    
    # Scale by Volume Intensity
    volume_ratio = data['volume'] / avg_volume_5day.replace(0, np.nan)
    final_signal = signal_with_divergence * volume_ratio
    
    # Clean and return
    final_signal = final_signal.replace([np.inf, -np.inf], np.nan)
    return final_signal
