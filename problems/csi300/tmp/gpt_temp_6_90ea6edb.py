import pandas as pd
from scipy.stats import skew

def heuristics_v2(df):
    rolling_median = df['close'].rolling(window=10, min_periods=1).median()
    return_skew = df['close'].pct_change().rolling(window=10, min_periods=1).apply(skew)
    price_ratio = (df['high'] / df['low']) * df['volume'].rolling(window=10, min_periods=1).mean()
    heuristics_matrix = rolling_median * return_skew / price_ratio
    return heuristics_matrix
