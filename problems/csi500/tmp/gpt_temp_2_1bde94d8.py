import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Classification
    short_term_vol = returns.rolling(3).std().ewm(span=5).mean()
    medium_term_vol = returns.rolling(10).std().ewm(span=10).mean()
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Regime Determination
    high_vol_regime = (volatility_ratio > 1.2).astype(int)
    low_vol_regime = (volatility_ratio < 0.8).astype(int)
    regime_indicator = high_vol_regime - low_vol_regime
    
    # Volatility-Adjusted Breakout Component
    rolling_high = df['high'].rolling(5).max()
    midpoint_price = (df['high'] + df['low']) / 2
    raw_breakout = (midpoint_price - rolling_high) / rolling_high
    
    # True Range and ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(5).mean()
    
    volatility_adjusted_breakout = raw_breakout / atr
    
    # Reversal Component with Volume Confirmation
    open_to_close_return = (df['close'] - df['open']) / df['open']
    prev_close_to_open_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    vol_adjusted_returns = returns / atr
    
    # Volume Confirmation
    volume_ma_20 = df['volume'].rolling(20).mean()
    volume_vs_ma = df['volume'] / volume_ma_20
    
    # Volume-Return Divergence
    volume_return_corr = df['volume'].rolling(10).corr(returns)
    volume_divergence_weight = 1 / (1 + abs(volume_return_corr))
    
    reversal_component = (open_to_close_return + prev_close_to_open_return + vol_adjusted_returns) / 3
    reversal_component = reversal_component * volume_divergence_weight * volume_vs_ma
    
    # Volume-Price Dynamics Integration
    # Volume Clustering Effect
    volume_ma_5 = df['volume'].rolling(5).mean()
    high_volume_periods = df['volume'] > 1.5 * volume_ma_5
    
    price_impact = (df['close'].diff() / df['volume']).where(high_volume_periods, 0)
    price_impact_per_cluster = price_impact.rolling(10).sum()
    
    volume_rank = df['volume'].rolling(20).apply(lambda x: pd.Series(x).rank().iloc[-1], raw=False)
    volume_clustering_score = price_impact_per_cluster * volume_rank
    
    # Volume-Price Timing
    hourly_volume_return_corr = df['volume'].rolling(6).corr(returns)
    timing_corr = df['volume'].rolling(10).corr(returns.shift(1))
    timing_score = hourly_volume_return_corr * timing_corr
    
    volume_price_component = (volume_clustering_score + timing_score) / 2
    
    # Regime-Adaptive Signal Combination
    # Component weighting by regime
    breakout_weight = high_vol_regime * 0.4 + low_vol_regime * 0.3
    reversal_weight = high_vol_regime * 0.4 + low_vol_regime * 0.5
    volume_price_weight = 0.2  # Same for both regimes
    
    # Composite signal
    composite_signal = (
        breakout_weight * volatility_adjusted_breakout +
        reversal_weight * reversal_component +
        volume_price_weight * volume_price_component
    )
    
    # Apply regime indicator and momentum overlay
    composite_signal = composite_signal * regime_indicator
    
    # Regime-specific momentum overlay
    momentum_high_vol = returns.rolling(1).mean()
    momentum_low_vol = returns.rolling(3).mean()
    
    momentum_overlay = high_vol_regime * momentum_high_vol + low_vol_regime * momentum_low_vol
    composite_signal = composite_signal * (1 + momentum_overlay)
    
    # Final Signal Processing
    # Remove extreme values using percentile-based clipping
    lower_bound = composite_signal.rolling(20).quantile(0.05)
    upper_bound = composite_signal.rolling(20).quantile(0.95)
    clipped_signal = composite_signal.clip(lower=lower_bound, upper=upper_bound)
    
    # Apply regime-dependent smoothing
    smoothed_high_vol = clipped_signal.ewm(span=2).mean()
    smoothed_low_vol = clipped_signal.ewm(span=5).mean()
    
    final_signal = high_vol_regime * smoothed_high_vol + low_vol_regime * smoothed_low_vol
    
    return final_signal
