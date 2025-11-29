import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Range Efficiency Calculation
    # Short-term: 3-day avg of (Close - Open) / (High - Low)
    daily_range_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    daily_range_efficiency = daily_range_efficiency.replace([np.inf, -np.inf], np.nan)
    short_term_range_eff = daily_range_efficiency.rolling(window=3, min_periods=2).mean()
    
    # Medium-term: 10-day avg of (High - Low) / Previous Close
    normalized_range = (data['high'] - data['low']) / data['close'].shift(1)
    normalized_range = normalized_range.replace([np.inf, -np.inf], np.nan)
    medium_term_range_eff = normalized_range.rolling(window=10, min_periods=8).mean()
    
    # Divergence Detection
    range_divergence = short_term_range_eff - medium_term_range_eff
    divergence_persistence = range_divergence.rolling(window=3, min_periods=2).apply(
        lambda x: np.sign(x).sum() if len(x) == 3 else np.nan
    )
    
    # Volume-Range Interaction
    amount_volume_ratio = data['amount'] / data['volume']
    amount_volume_ratio = amount_volume_ratio.replace([np.inf, -np.inf], np.nan)
    vol_range_ratio = amount_volume_ratio / (data['high'] - data['low'])
    vol_range_ratio = vol_range_ratio.replace([np.inf, -np.inf], np.nan)
    vol_range_divergence = vol_range_ratio - vol_range_ratio.rolling(window=5, min_periods=3).mean()
    
    # Intraday Price Memory
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    range_context_gap = opening_gap / (data['high'] - data['low'])
    range_context_gap = range_context_gap.replace([np.inf, -np.inf], np.nan)
    
    previous_day_influence = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    previous_day_influence = previous_day_influence.replace([np.inf, -np.inf], np.nan)
    
    # Pressure Enhancement
    buying_pressure = (data['close'] - data['low']) / (data['high'] - data['low'])
    buying_pressure = buying_pressure.replace([np.inf, -np.inf], np.nan)
    
    range_weighted_pressure = buying_pressure * short_term_range_eff
    pressure_momentum = range_weighted_pressure - range_weighted_pressure.rolling(window=3, min_periods=2).mean()
    
    # Composite Signal Generation
    composite_signal = (
        0.3 * range_divergence.fillna(0) +
        0.2 * divergence_persistence.fillna(0) +
        0.15 * vol_range_divergence.fillna(0) +
        0.15 * range_context_gap.fillna(0) +
        0.1 * previous_day_influence.fillna(0) +
        0.1 * pressure_momentum.fillna(0)
    )
    
    # Cross-sectional ranking
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    # Apply cross-sectional ranking daily
    ranked_signal = composite_signal.groupby(composite_signal.index).transform(cross_sectional_rank)
    
    return ranked_signal
