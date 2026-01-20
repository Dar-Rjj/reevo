import pandas as pd
import numpy as np

def heuristics_v2(df):
    price_changes = df[['open', 'high', 'low', 'close']].pct_change().abs()
    volatility = price_changes.std(axis=1)
    weighted_changes = (price_changes['close'] * volatility) / (df['volume'] + 1e-6)
    heuristics_matrix = weighted_changes.rolling(5).apply(lambda x: np.prod(x + 1)**(1/len(x)) - 1)
    return heuristics_matrix
