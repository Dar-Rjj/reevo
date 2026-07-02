import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 1. Ultra-short momentum (1-day % change) for reactivity
    momentum = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-7)
    
    # 2. Volume acceleration (current vs prior day ratio)
    volume_accel = df['volume'] / (df['volume'].shift(1) + 1e-7)
    
    # 3. Intraday pressure (close position in daily range)
    intraday_pressure = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # 4. Price-range volatility scaling
    daily_range = df['high'] - df['low']
    range_volatility = daily_range / (daily_range.rolling(5).mean() + 1e-7)
    
    # 5. Multiplicative combination with volatility dampening
    raw_factor = momentum * np.log1p(volume_accel) * intraday_pressure
    volatility_adjusted = raw_factor / (range_volatility + 1e-7)
    
    # 6. Hamming-window smoothing (5-period)
    window_size = 5
    hamming_weights = np.hamming(window_size)
    smoothed_factor = volatility_adjusted.rolling(window_size).apply(
        lambda x: np.sum(x * hamming_weights / hamming_weights.sum())
    )
    
    # 7. Cross-sectional rank normalization (0-1 scaling)
    ranked_factor = smoothed_factor.rank(pct=True)
    
    return ranked_factor
