import pandas as pd
import numpy as np

def heuristics_v2(df):
    relative_strength = (df['close'] - df['low']) / (df['high'] - df['low'])
    volume_weighted = df['volume'] * relative_strength
    acceleration = volume_weighted.diff().rolling(5).mean()
    heuristics_matrix = acceleration.rolling(10).apply(lambda x: len(x) / np.sum(1 / x))
    return heuristics_matrix
