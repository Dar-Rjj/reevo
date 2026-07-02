import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 3-day rolling z-score for close prices
    rolling_mean = df['close'].rolling(window=3, min_periods=1).mean()
    rolling_std = df['close'].rolling(window=3, min_periods=1).std()
    z_score = (df['close'] - rolling_mean) / rolling_std
    
    # Calculate 10-day volume percentile
    volume_percentile = df['volume'].rolling(window=10, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Normalize z-score by volume activity
    normalized_z_score = z_score / volume_percentile
    
    # Compute bid-ask proxy (High - Low)/Close * Volume
    bid_ask_proxy = ((df['high'] - df['low']) / df['close']) * df['volume']
    
    # Weighted reversion score: multiply z-score by liquidity
    weighted_score = normalized_z_score * bid_ask_proxy
    
    # Apply sign correction: positive = overbought, negative = oversold
    final_factor = -weighted_score
    
    return final_factor
