import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Relative Strength calculation
    # ratio = high / rolling_mean(low, window=15)
    rolling_low = df['low'].rolling(window=15, min_periods=1).mean()
    ratio = df['high'] / rolling_low
    
    # Normalize with cross-sectional rank
    normalized_rs = ratio.groupby(ratio.index).rank(pct=True)
    
    # Skew Adjustment calculation
    # rolling_skewness of 3-day close returns over 10-day window
    delta_close = df['close'].pct_change(periods=3)
    rolling_skew = delta_close.rolling(window=10, min_periods=1).skew()
    
    # EMA decay with alpha=0.2 and window=7
    def ema_decay(series):
        return series.ewm(alpha=0.2, adjust=False, min_periods=1).mean()
    
    decayed_skew = rolling_skew.groupby(rolling_skew.index).apply(ema_decay)
    
    # Multiply Relative Strength with decayed skew adjustment
    factor = normalized_rs * decayed_skew
    
    return factor
