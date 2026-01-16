import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Measure Intraday Range Extent
    intraday_range = (df['high'] - df['low']) / df['open']
    
    # Normalize Intraday Momentum with rolling 5-day mean
    rolling_mean_intraday = intraday_range.rolling(window=5, min_periods=1).mean()
    normalized_intraday_momentum = intraday_range / rolling_mean_intraday
    
    # Identify Extreme Range
    extreme_momentum_flag = np.where(
        (normalized_intraday_momentum > 1.5) | (normalized_intraday_momentum < 0.5),
        normalized_intraday_momentum,
        np.nan
    )
    
    # Volume Surge Ratio
    volume_20ma = df['volume'].rolling(window=20, min_periods=1).mean()
    volume_surge_ratio = df['volume'] / volume_20ma
    
    # Scale Momentum by Volume Surge
    scaled_momentum = extreme_momentum_flag * volume_surge_ratio
    
    # Detect Breakout with Momentum and Volume
    breakout_signal = scaled_momentum.rolling(window=20, min_periods=1).apply(
        lambda x: (x[-1] - x[:-1].mean()) / x[:-1].std() if x[:-1].std() != 0 else np.nan
    )
    
    # Confirm Breakout Direction
    five_day_high = df['high'].rolling(window=5, min_periods=1).max()
    five_day_low = df['low'].rolling(window=5, min_periods=1).min()
    
    breakout_direction = np.where(
        df['close'] > five_day_high,
        1,
        np.where(df['close'] < five_day_low, -1, 0)
    )
    
    # Final Factor Calculation
    factor = breakout_signal * breakout_direction
    
    return factor
