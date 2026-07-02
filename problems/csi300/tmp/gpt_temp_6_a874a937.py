import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Liquidity Change Signal
    # Delta Volume over a window of 5 days
    delta_volume = df['volume'].diff().rolling(window=5, min_periods=1).mean()
    
    # Rolling rank of delta volume over a window of 20 days
    rolling_rank = delta_volume.rolling(window=20, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Z-score of cross-sectional rank
    zscore = (rolling_rank - rolling_rank.mean()) / rolling_rank.std()
    
    # Volume-Adjusted Momentum Confirmation
    # EMA of close prices over a span of 10 days
    ema_close = df['close'].ewm(span=10, adjust=False).mean()
    
    # Rolling mean of volume over a window of 10 days
    rolling_mean_volume = df['volume'].rolling(window=10, min_periods=1).mean()
    
    # Ratio of EMA of close to rolling mean of volume
    ratio = ema_close / rolling_mean_volume
    
    # Volatility-adjusted return
    # Return of close prices over a window of 1 day
    ret = df['close'].pct_change()
    
    # Rolling standard deviation of returns over a window of 10 days
    rolling_std = ret.rolling(window=10, min_periods=1).std()
    
    # Volatility-adjusted return
    volatility_adjusted_return = ret / rolling_std
    
    # Combine signals
    factor = zscore * ratio * volatility_adjusted_return
    
    return factor
