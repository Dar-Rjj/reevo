import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Fractal Regime Dynamics Factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Helper function to detect local extrema (fractal points)
    def detect_local_extrema(series, window=5):
        """Detect local maxima and minima within a rolling window"""
        maxima = (series == series.rolling(window, center=True).max()) & (series.rolling(window, center=True).count() == window)
        minima = (series == series.rolling(window, center=True).min()) & (series.rolling(window, center=True).count() == window)
        return maxima | minima
    
    # Calculate fractal-related features
    # 1. High-Low Fractal Complexity
    high_low_range = data['high'] - data['low']
    high_extrema = detect_local_extrema(data['high'])
    low_extrema = detect_local_extrema(data['low'])
    fractal_count = (high_extrema | low_extrema).rolling(5).sum()
    fractal_complexity = fractal_count / (high_low_range + 1e-8)
    
    # 2. Volume-Fractal Regime Alignment
    volume_rolling = data['volume'].rolling(20)
    volume_clustering = (data['volume'] - volume_rolling.mean()) / (volume_rolling.std() + 1e-8)
    fractal_alignment = fractal_complexity * volume_clustering
    
    # 3. Fractal Persistence Ratio
    fractal_5day = fractal_count
    fractal_20day = (high_extrema | low_extrema).rolling(20).sum()
    fractal_persistence = fractal_5day / (fractal_20day + 1e-8)
    
    # 4. Fractal Formation in High-Flow Regimes
    flow_regime = data['amount'].rolling(10).mean()
    fractal_formation = (high_low_range * fractal_count) / (data['volume'] + 1e-8)
    fractal_formation_flow = fractal_formation * flow_regime
    
    # 5. Volume Acceleration at Fractal Boundaries
    volume_spike = data['volume'] / data['volume'].rolling(10).mean()
    volume_acceleration = volume_spike * fractal_complexity
    
    # 6. Fractal Completion Timing
    fractal_timing = fractal_count.diff(5).abs() / (fractal_count.rolling(10).std() + 1e-8)
    
    # 7. Fractal Breakpoint Flow
    close_open_range = (data['close'] - data['open']).abs()
    fractal_completion = fractal_count.diff().clip(lower=0)
    breakpoint_flow = (close_open_range * fractal_completion) / (high_low_range + 1e-8)
    
    # 8. Volume-Weighted Regime Transition
    regime_shift = fractal_complexity.diff(5).abs()
    volume_weighted_transition = regime_shift * volume_clustering
    
    # 9. Cross-Regime Fractal Momentum
    returns = data['close'].pct_change()
    fractal_returns = returns.rolling(5).mean() * fractal_complexity
    cross_regime_momentum = fractal_returns.rolling(10).sum()
    
    # 10. Intraday Fractal-Regime Coherence
    morning_strength = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    afternoon_strength = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    intraday_coherence = (morning_strength * afternoon_strength) * fractal_complexity
    
    # 11. Fractal Persistence Across Volatility Regimes
    volatility = data['close'].pct_change().rolling(20).std()
    fractal_stability = fractal_count.rolling(10).std()
    volatility_persistence = fractal_stability / (volatility + 1e-8)
    
    # 12. Volume-Fractal Regime Efficiency
    regime_efficiency = (close_open_range * fractal_complexity) / (data['volume'] + 1e-8)
    
    # Combine all components into final factor
    factor_components = [
        fractal_complexity.rank(pct=True),
        fractal_alignment.rank(pct=True),
        fractal_persistence.rank(pct=True),
        fractal_formation_flow.rank(pct=True),
        volume_acceleration.rank(pct=True),
        fractal_timing.rank(pct=True),
        breakpoint_flow.rank(pct=True),
        volume_weighted_transition.rank(pct=True),
        cross_regime_momentum.rank(pct=True),
        intraday_coherence.rank(pct=True),
        volatility_persistence.rank(pct=True),
        regime_efficiency.rank(pct=True)
    ]
    
    # Equal-weighted combination with cross-sectional normalization
    final_factor = pd.concat(factor_components, axis=1).mean(axis=1)
    
    # Cross-sectional z-score normalization
    def cross_sectional_zscore(series):
        return (series - series.mean()) / (series.std() + 1e-8)
    
    final_factor = final_factor.groupby(final_factor.index).transform(cross_sectional_zscore)
    
    return final_factor
