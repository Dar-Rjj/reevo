import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Intraday Momentum Strength
    # Intraday Return
    intraday_return = data['close'] / data['open'] - 1
    
    # Absolute Intraday Movement
    abs_intraday_movement = np.abs(data['close'] - data['open'])
    
    # Momentum Persistence Signal with cubic transformation
    momentum_persistence = intraday_return * abs_intraday_movement
    momentum_persistence = momentum_persistence ** 3
    
    # 2. Calculate Volume Confirmation Pattern
    # Current Day Volume
    current_volume = data['volume']
    
    # Rolling Median Volume (15-day)
    rolling_median_volume = current_volume.rolling(window=15, min_periods=1).median()
    
    # Volume Confirmation Score with hyperbolic tangent
    volume_ratio = current_volume / rolling_median_volume
    volume_confirmation = np.tanh(volume_ratio)
    
    # 3. Calculate Price Continuation Component
    # Prior Day Return
    prior_day_return = data['close'].shift(1) / data['close'].shift(2) - 1
    
    # Prior Day Volume Ratio
    prior_day_volume_ratio = data['volume'].shift(1) / data['volume'].shift(2)
    
    # Trend Consistency with exponential weighting
    trend_consistency = prior_day_return * prior_day_volume_ratio
    # Apply exponential decay with half-life of 3 days
    trend_consistency = trend_consistency.ewm(halflife=3, min_periods=1).mean()
    
    # 4. Combine All Components
    # Multiply components
    combined_factor = momentum_persistence * volume_confirmation * trend_consistency
    
    # Apply cross-sectional z-score normalization
    def cross_sectional_zscore(series):
        return (series - series.mean()) / series.std()
    
    # Calculate cross-sectional z-score for each day
    alpha_factor = combined_factor.groupby(level='date').apply(cross_sectional_zscore)
    
    return alpha_factor
