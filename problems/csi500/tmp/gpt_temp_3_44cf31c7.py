import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Momentum Acceleration & Liquidity Gradient Alpha Factor
    
    Combines intraday momentum curvature, multi-timeframe acceleration patterns,
    and liquidity gradient analysis to generate cross-sectional alpha signals.
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate intraday momentum curvature components
    # Early session acceleration (first hour proxy - using open to high/low ratios)
    early_acceleration = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    
    # Late session momentum divergence (using close relative to daily range)
    daily_range = data['high'] - data['low']
    late_momentum = (data['close'] - (data['low'] + 0.5 * daily_range)) / ((data['high'] - 0.5 * daily_range) - data['close'] + 1e-8)
    
    # Multi-timeframe acceleration
    # Daily returns
    daily_returns = data['close'].pct_change()
    
    # Momentum acceleration (current return vs previous return)
    momentum_acceleration = daily_returns / (daily_returns.shift(1) + 1e-8)
    
    # Liquidity gradient analysis
    # Volume flow concentration (using rolling volume patterns)
    volume_ma_short = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_ma_long = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_gradient = (volume_ma_short - volume_ma_long) / (volume_ma_long + 1e-8)
    
    # Price-volume gradient divergence
    price_momentum = data['close'].pct_change(periods=5)
    volume_momentum = data['volume'].pct_change(periods=5)
    pv_divergence = price_momentum - volume_momentum
    
    # Cross-sectional normalization components
    def cross_sectional_rank(series):
        """Calculate cross-sectional rank within each period"""
        return series.rank(pct=True)
    
    # Calculate cross-sectional ranks for key components
    early_accel_rank = early_acceleration.groupby(early_acceleration.index).transform(cross_sectional_rank)
    late_momentum_rank = late_momentum.groupby(late_momentum.index).transform(cross_sectional_rank)
    momentum_accel_rank = momentum_acceleration.groupby(momentum_acceleration.index).transform(cross_sectional_rank)
    volume_gradient_rank = volume_gradient.groupby(volume_gradient.index).transform(cross_sectional_rank)
    pv_divergence_rank = pv_divergence.groupby(pv_divergence.index).transform(cross_sectional_rank)
    
    # Acceleration regime classification
    # High acceleration regime indicator
    high_accel_threshold = momentum_acceleration.quantile(0.7)
    high_accel_regime = (momentum_acceleration > high_accel_threshold).astype(int)
    
    # Liquidity gradient regime
    high_liquidity_threshold = volume_gradient.quantile(0.7)
    high_liquidity_regime = (volume_gradient > high_liquidity_threshold).astype(int)
    
    # Regime-dependent weighting
    # High acceleration + high liquidity regime (optimal)
    optimal_regime = high_accel_regime * high_liquidity_regime
    
    # High acceleration + low liquidity (unsustainable)
    unsustainable_regime = high_accel_regime * (1 - high_liquidity_regime)
    
    # Low acceleration + high liquidity (accumulation)
    accumulation_regime = (1 - high_accel_regime) * high_liquidity_regime
    
    # Gradient persistence analysis
    # Acceleration persistence (rolling correlation of acceleration)
    accel_persistence = momentum_acceleration.rolling(window=10, min_periods=5).apply(
        lambda x: x.corr(pd.Series(range(len(x)), index=x.index)) if len(x) > 1 else 0
    )
    
    # Liquidity gradient persistence
    liquidity_persistence = volume_gradient.rolling(window=10, min_periods=5).apply(
        lambda x: x.corr(pd.Series(range(len(x)), index=x.index)) if len(x) > 1 else 0
    )
    
    # Integrated gradient alpha construction
    # Core momentum acceleration score
    momentum_score = (
        0.4 * early_accel_rank + 
        0.3 * late_momentum_rank + 
        0.3 * momentum_accel_rank
    )
    
    # Core liquidity gradient score
    liquidity_score = (
        0.6 * volume_gradient_rank + 
        0.4 * pv_divergence_rank
    )
    
    # Regime-enhanced combination
    # Optimal regime: emphasize both momentum and liquidity
    optimal_weight = 1.2
    # Unsustainable regime: penalize momentum without liquidity
    unsustainable_weight = 0.5
    # Accumulation regime: emphasize liquidity over momentum
    accumulation_weight = 0.8
    # Other regimes: neutral weighting
    neutral_weight = 1.0
    
    # Apply regime weights
    regime_weighted_momentum = (
        optimal_regime * optimal_weight * momentum_score +
        unsustainable_regime * unsustainable_weight * momentum_score +
        accumulation_regime * accumulation_weight * momentum_score +
        (1 - optimal_regime - unsustainable_regime - accumulation_regime) * neutral_weight * momentum_score
    )
    
    regime_weighted_liquidity = (
        optimal_regime * optimal_weight * liquidity_score +
        unsustainable_regime * unsustainable_weight * liquidity_score +
        accumulation_regime * accumulation_weight * liquidity_score +
        (1 - optimal_regime - unsustainable_regime - accumulation_regime) * neutral_weight * liquidity_score
    )
    
    # Persistence adjustment
    persistence_adjustment = (accel_persistence + liquidity_persistence) / 2
    persistence_weight = 0.3 + 0.7 * persistence_adjustment  # Base 0.3, up to 1.0
    
    # Final integrated gradient alpha
    integrated_alpha = (
        persistence_weight * (0.6 * regime_weighted_momentum + 0.4 * regime_weighted_liquidity)
    )
    
    # Cross-sectional normalization of final alpha
    final_factor = integrated_alpha.groupby(integrated_alpha.index).transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    # Fill NaN values with 0 (neutral signal)
    final_factor = final_factor.fillna(0)
    
    return final_factor
