import pandas as pd
import numpy as np

def heuristics_v2(df):
    log_price_change = np.log(df['close'] / df['close'].shift(1))
    volatility = (df['high'] - df['low']) / df['close'].shift(1)
    volume_change = df['volume'] - df['volume'].shift(5)
    heuristics_matrix = (log_price_change * volatility).rolling(10).apply(lambda x: len(x) / np.sum(1/x)) * volume_change
    return heuristics_matrix
