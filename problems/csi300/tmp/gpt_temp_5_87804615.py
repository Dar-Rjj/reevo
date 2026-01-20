import pandas as pd
import numpy as np

def heuristics_v2(df):
    weights = {'open': 0.2, 'high': 0.3, 'low': 0.1, 'close': 0.3, 'volume': 0.1}
    heuristics_matrix = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        factor_value = 0
        for feature, weight in weights.items():
            factor_value += df.loc[date, feature] * weight
        heuristics_matrix[date] = factor_value
    
    heuristics_matrix = heuristics_matrix.ewm(span=5, adjust=False).mean()
    return heuristics_matrix
