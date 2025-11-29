import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate intraday momentum strength
    df = df.copy()
    
    # Compute price change from open
    price_change = df['close'] - df['open']
    
    # Calculate true range
    prev_close = df['close'].shift(1)
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - prev_close)
    low_prev_close = abs(df['low'] - prev_close)
    true_range = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    
    # Intraday momentum strength
    momentum_strength = price_change / true_range
    momentum_strength = momentum_strength.replace([np.inf, -np.inf], np.nan)
    
    # Calculate daily momentum signs
    momentum_sign = np.sign(momentum_strength)
    
    # Calculate 5-day trend direction consistency (autocorrelation of signs)
    trend_consistency = momentum_sign.rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr() if len(x) >= 3 else np.nan, raw=False
    )
    
    # Calculate volume ratio (current volume / 20-day average volume)
    volume_avg_20d = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = df['volume'] / volume_avg_20d
    
    # Combine components
    factor = momentum_strength * trend_consistency * volume_ratio
    
    return factor
