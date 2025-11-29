import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    prev_close = data['close'].shift(1)
    method1 = data['high'] - data['low']
    method2 = abs(data['high'] - prev_close)
    method3 = abs(data['low'] - prev_close)
    true_range = pd.concat([method1, method2, method3], axis=1).max(axis=1)
    true_range = true_range.replace(0, np.nan)  # Handle zero division cases
    
    # Calculate Range Efficiency Ratio
    daily_range = data['high'] - data['low']
    range_efficiency_ratio = daily_range / true_range
    range_efficiency_ratio = range_efficiency_ratio.fillna(0)
    
    # Calculate Momentum Component
    midpoint = (data['high'] + data['low']) / 2
    range_efficiency_momentum = (midpoint - prev_close) * range_efficiency_ratio
    
    # Calculate Volume Momentum
    volume_median_3d = data['volume'].rolling(window=3, min_periods=1).median()
    volume_momentum = data['volume'] / volume_median_3d
    volume_momentum = volume_momentum.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Calculate Volume-Range Divergence
    range_eff_percentile = range_efficiency_ratio.rolling(window=5, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    volume_mom_percentile = volume_momentum.rolling(window=5, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Detect divergence conditions
    high_range_low_volume = (range_eff_percentile > 0.7) & (volume_mom_percentile < 0.3)
    low_range_high_volume = (range_eff_percentile < 0.3) & (volume_mom_percentile > 0.7)
    
    # Calculate divergence strength
    divergence_strength = abs(range_eff_percentile - volume_mom_percentile)
    
    # Generate Volume Adjustment Factor
    vol_std_10d = data['volume'].rolling(window=10, min_periods=1).std()
    vol_adj_base = volume_momentum / vol_std_10d.replace(0, np.nan).fillna(1)
    
    # Apply conditional logic for volume adjustment
    volume_adjustment = vol_adj_base.copy()
    volume_adjustment[high_range_low_volume] = -vol_adj_base[high_range_low_volume] * divergence_strength[high_range_low_volume]
    volume_adjustment[low_range_high_volume] = -vol_adj_base[low_range_high_volume] * divergence_strength[low_range_high_volume]
    
    # Combine Range Efficiency Momentum with Volume Confirmation
    composite_factor = range_efficiency_momentum * volume_adjustment
    
    # Incorporate Volatility Persistence
    daily_ranges = data['high'] - data['low']
    range_std_5d = daily_ranges.rolling(window=5, min_periods=1).std()
    range_std_20d = daily_ranges.rolling(window=20, min_periods=1).std()
    volatility_persistence = range_std_5d / range_std_20d.replace(0, np.nan).fillna(1)
    
    # Scale final factor by volatility persistence
    final_factor = composite_factor * volatility_persistence
    
    return final_factor
