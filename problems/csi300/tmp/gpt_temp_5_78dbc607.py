import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Trend curvature: 2nd derivative of log prices
    log_prices = np.log(df['close'])
    trend_velocity = log_prices.diff(1)
    trend_acceleration = trend_velocity.diff(1)
    trend_curvature = trend_acceleration - trend_velocity.rolling(3).mean()
    
    # Dynamic volume-pressure: volume relative to recent range
    range_5d = (df['high'] - df['low']).rolling(5).mean()
    volume_pressure = df['volume'] / (range_5d * df['close'] + 1e-7)
    
    # Relative position strength: close vs intraday range
    normalized_close = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    position_strength = normalized_close - normalized_close.rolling(3).mean()
    
    # Composite factor: curvature × volume-pressure × position strength
    factor = trend_curvature * volume_pressure * position_strength
    
    # Robust smoothing using 3-day trimmed mean
    factor = factor.rolling(3, center=True).apply(
        lambda x: np.mean(np.sort(x)[1:-1]) if len(x) == 3 else np.nan
    )
    
    return factor
