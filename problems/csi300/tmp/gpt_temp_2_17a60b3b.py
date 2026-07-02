import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    # ratio = close_t / rolling_mean(close_{t-10}, window=10)
    rolling_mean_close = df['close'].shift(1).rolling(window=10).mean()
    ratio = df['close'] / rolling_mean_close
    
    # normalize using cross-sectional rank
    normalized_ratio = ratio.rank(axis=0, pct=True)
    
    # Volume Divergence
    # rolling_rank of volume over 20 days
    rolling_rank_volume = df['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # delta of rolling_rank_volume over 5 days
    delta_volume_rank = rolling_rank_volume.diff(5)
    
    # EMA decay with alpha=0.3 and span=5
    def ema_decay(series):
        return series.ewm(alpha=0.3, adjust=False).mean()
    
    decayed_delta = delta_volume_rank.groupby(level=0).apply(ema_decay)
    
    # Multiply normalized ratio with decayed delta
    factor = normalized_ratio * decayed_delta
    
    return factor
