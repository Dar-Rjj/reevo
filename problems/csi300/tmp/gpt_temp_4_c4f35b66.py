import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Trend Momentum Component
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    delta = ema_10.diff(5)
    rolling_std = df['close'].rolling(20).std()
    normalized_trend = delta / (rolling_std + 1e-6)
    
    # Liquidity Weighting components
    # Volume normalization
    rolling_vol_mean = df['volume'].rolling(20).mean()
    normalized_vol = df['volume'] / (rolling_vol_mean + 1e-6)
    
    # Spread adjustment
    spread_ratio = (df['high'] - df['low']) / (df['close'] + 1e-6)
    vol_rank = df['volume'].rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combine components
    spread_adjusted = spread_ratio * vol_rank
    weighted_trend = normalized_trend * normalized_vol * spread_adjusted
    
    # Apply decay factor
    decay_factor = weighted_trend.ewm(span=10, adjust=False).mean()
    
    return decay_factor
