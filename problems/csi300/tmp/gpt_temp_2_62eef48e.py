import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-adjusted momentum
    vol = df['close'].rolling(window=10).std()
    ret = df['close'].pct_change(5)
    adj_momentum = ret / (vol + 1e-6)
    
    # Liquidity-weighted mean reversion
    liq = np.log(df['volume'] * df['close'])
    rolling_median = df['close'].rolling(window=20).median()
    deviation = (df['close'] - rolling_median) / rolling_median
    weighted_reversion = deviation * liq.ewm(span=10).mean()
    
    # Combine components with dynamic weighting
    heuristics_matrix = adj_momentum.ewm(span=5).mean() - weighted_reversion.rolling(window=10).apply(lambda x: np.percentile(x, 30))
    
    return heuristics_matrix
