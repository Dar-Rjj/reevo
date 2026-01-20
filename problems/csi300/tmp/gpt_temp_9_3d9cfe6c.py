import pandas as pd
import numpy as np

def heuristics_v2(df):
    heuristics_matrix = (
        (df['close'].rolling(window=10).apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))) * 
        (df['close'].pct_change(periods=5).rolling(window=5).mean()) +
        (df['close'].shift(3) / df['close'].shift(1) - 1) * df['volume'].rolling(window=3).mean()
    )
    return heuristics_matrix
