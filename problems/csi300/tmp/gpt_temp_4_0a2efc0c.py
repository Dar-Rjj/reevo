import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Impact Signal
    # ratio = high_t / low_t
    ratio = df['high'] / df['low']
    
    # rolling rank of ratio over 10 days
    rolling_rank = ratio.rolling(window=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # normalize the rolling rank
    price_impact_signal = (rolling_rank - rolling_rank.mean()) / rolling_rank.std()
    
    # Volume Divergence
    # EMA of volume with span=10
    ema_volume = df['volume'].ewm(span=10, adjust=False).mean()
    
    # difference between current volume and EMA volume
    volume_diff = df['volume'] - ema_volume
    
    # cross-sectional zscore of volume difference
    volume_zscore = (volume_diff - volume_diff.mean()) / volume_diff.std()
    
    # Combine signals
    factor = price_impact_signal + volume_zscore
    
    return factor
