import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Inefficiency branch
    # Calculate rolling mean of high and low
    rolling_mean_hl = (df['high'].rolling(window=10, min_periods=1).mean() + 
                       df['low'].rolling(window=10, min_periods=1).mean()) / 2
    
    # Calculate absolute ratio of close to rolling mean
    abs_ratio = np.abs(df['close'] / rolling_mean_hl - 1)
    
    # Normalize with rolling rank
    price_inefficiency = abs_ratio.rolling(window=10, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Liquidity Imbalance branch
    # Calculate ratio of volume to rolling mean volume
    rolling_mean_vol = df['volume'].rolling(window=20, min_periods=1).mean()
    vol_ratio = df['volume'] / rolling_mean_vol
    
    # Apply EMA decay
    def ema_decay(series, alpha=0.3, window=5):
        return series.ewm(alpha=alpha, adjust=False).mean().shift(1).rolling(window=window, min_periods=1).mean()
    
    liquidity_imbalance = vol_ratio * ema_decay(price_inefficiency)
    
    # Combine both branches
    factor = price_inefficiency * liquidity_imbalance
    
    return factor
