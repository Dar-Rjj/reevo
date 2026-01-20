import pandas as pd
import numpy as np

def heuristics_v2(df):
    """{The new algorithm generates alpha factors by computing the geometric mean of price features, adjusted by the exponential moving average of volume changes over a rolling window.}"""
    features = ['open', 'high', 'low', 'close']
    geom_mean = df[features].apply(lambda x: np.exp(np.log(x).mean()), axis=1)
    vol_change = df['volume'].diff().abs().ewm(span=5).mean()
    heuristics_matrix = geom_mean * vol_change
    return heuristics_matrix
