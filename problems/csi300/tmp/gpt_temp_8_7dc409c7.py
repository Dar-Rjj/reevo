import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    price_volatility = df['close'].rolling(window=5).std()
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(len(x)), x)[0])
    heuristics_matrix = price_volatility * volume_trend
    return heuristics_matrix
