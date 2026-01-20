import pandas as pd

def heuristics_v2(df):
    price_median = df['close'].rolling(window=5).median()
    volume_mad = df['volume'].rolling(window=5).apply(lambda x: (x - x.mean()).abs().mean())
    heuristics_matrix = (df['close'] - price_median) / volume_mad
    return heuristics_matrix
