import pandas as pd
import numpy as np

def heuristics_v2(df):
    {
        "Combines normalized price volatility, volume acceleration, and mean-reversion signal with a rolling geometric mean and outlier trimming."
    }
    price_volatility = df['close'].rolling(5).std() / df['close'].rolling(10).mean()
    volume_accel = df['volume'].diff(3) / df['volume'].rolling(5).std()
    mean_reversion = -df['close'].pct_change(5)
    heuristics_matrix = np.sqrt(np.abs(price_volatility * volume_accel * mean_reversion)).rolling(8).apply(lambda x: np.percentile(x, 75))
    return heuristics_matrix
