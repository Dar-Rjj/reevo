import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Calculate Normalized Momentum (Price Change from Open / True Range)
    price_change = df['close'] - df['open']
    normalized_momentum = price_change / true_range.replace(0, np.nan)
    
    # Calculate Rolling Trend Consistency (5-day autocorrelation of momentum direction)
    momentum_sign = np.sign(normalized_momentum)
    trend_consistency = momentum_sign.rolling(window=5, min_periods=3).apply(
        lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
    )
    
    # Multiply by current trend strength
    trend_persistence = trend_consistency * normalized_momentum
    
    # Calculate Volume Participation (current volume / 20-day average volume)
    volume_ratio = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    
    # Final factor: Volume-weighted trend persistence
    factor = trend_persistence * volume_ratio
    
    return factor
