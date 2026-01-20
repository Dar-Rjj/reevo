import pandas as pd
import numpy as np

def heuristics_v2(df):
    heuristics_matrix = np.exp(df['close'].pct_change(3).rolling(3).mean()) * (df['high'].rolling(5).apply(lambda x: x.max() - x.min()) + df['low'].rolling(6).apply(lambda x: x.max() - x.min())) / df['volume'].rolling(4).mean()
    return heuristics_matrix
