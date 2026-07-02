import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    {The algorithm combines mean-reversion adjusted by volume spike detection, using rolling percentile filters for outlier mitigation.}
    """
    mean_reversion = (df['close'].shift(5) - df['close']) / df['close'].shift(5)
    volume_spike = df['volume'] / df['volume'].rolling(10).max()
    heuristics_matrix = mean_reversion.rolling(15).apply(lambda x: np.percentile(x, 30)) * volume_spike.ewm(span=5).mean()
    return heuristics_matrix
