import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Momentum Divergence Alpha Factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Directional Volume Momentum Analysis
    # Short-Term Price Momentum (3-day Close-to-Close)
    price_momentum = data['close'].pct_change(periods=3)
    
    # Volume Momentum Acceleration (3-day volume change rate)
    volume_momentum = data['volume'].pct_change(periods=3)
    
    # Price-Volume Divergence Ratio
    divergence_ratio = price_momentum / (volume_momentum + 1e-8)
    
    # Divergence Persistence (5-day consistency)
    divergence_sign = np.sign(divergence_ratio)
    divergence_persistence = divergence_sign.rolling(window=5, min_periods=3).sum() / 5
    divergence_persistence_strength = divergence_persistence.abs()
    
    # Momentum Divergence Signal
    momentum_divergence = divergence_ratio * divergence_persistence_strength
    
    # 2. Intraday Price Range Efficiency
    # Opening Range Efficiency (Open to High utilization)
    opening_range = (data['high'] - data['open']) / (data['open'] + 1e-8)
    
    # Closing Range Compression (Low to Close efficiency)
    closing_range = (data['close'] - data['low']) / (data['low'] + 1e-8)
    
    # Range Efficiency-Volume Integration
    volume_weighted_opening = opening_range * data['volume']
    volume_weighted_closing = closing_range * data['volume']
    
    # Range Expansion Signals
    range_efficiency_ratio = volume_weighted_opening / (volume_weighted_closing + 1e-8)
    
    # 3. Price Gap Momentum Persistence
    # Daily Opening Gaps
    daily_gaps = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Gap Momentum Clustering (3-day gap direction persistence)
    gap_direction = np.sign(daily_gaps)
    gap_persistence = gap_direction.rolling(window=3, min_periods=2).sum() / 3
    
    # Gap-Volume Interaction
    gap_volume_confirmation = daily_gaps.abs() * data['volume']
    
    # Gap Momentum Signals
    gap_momentum = gap_persistence * gap_volume_confirmation
    
    # 4. Volume Concentration Asymmetry
    # Calculate daily returns
    daily_returns = data['close'].pct_change()
    
    # Up-Day Volume Concentration
    up_days = daily_returns > 0
    up_day_volume = data['volume'].where(up_days, 0)
    total_up_volume = up_day_volume.rolling(window=5, min_periods=3).sum()
    
    # Down-Day Volume Intensity
    down_days = daily_returns < 0
    down_day_volume = data['volume'].where(down_days, 0)
    total_down_volume = down_day_volume.rolling(window=5, min_periods=3).sum()
    
    # Volume Directional Bias
    volume_bias = total_up_volume / (total_down_volume + 1e-8)
    
    # Volume Asymmetry Momentum (5-day change rate)
    asymmetry_momentum = volume_bias.pct_change(periods=5)
    
    # Volume Asymmetry Signal
    volume_asymmetry = volume_bias * asymmetry_momentum
    
    # 5. Composite Momentum Divergence Alpha
    # Core Price-Volume Divergence Component
    core_divergence = momentum_divergence * range_efficiency_ratio * gap_momentum
    
    # Volume Asymmetry Enhancement
    enhanced_divergence = core_divergence * volume_asymmetry
    
    # Final Alpha Integration with cross-sectional ranking
    # Remove extreme values and normalize
    alpha_factor = enhanced_divergence.replace([np.inf, -np.inf], np.nan)
    
    # Cross-sectional z-score normalization
    def cross_sectional_zscore(series):
        return (series - series.mean()) / (series.std() + 1e-8)
    
    # Apply cross-sectional normalization
    alpha_factor = alpha_factor.groupby(alpha_factor.index).transform(cross_sectional_zscore)
    
    return alpha_factor
