import pandas as pd
import numpy as np

def heuristics_v2(df):
    log_returns = np.log(df['close'] / df['close'].shift(1))
    volume_std = df['volume'].rolling(14).std()
    custom_avg = (log_returns * volume_std).rolling(8).apply(lambda x: (x.iloc[:4].sum() + 2 * x.iloc[4:].sum()) / 10, raw=False)
    heuristics_matrix = custom_avg.ewm(span=6, adjust=False).mean()
    return heuristics_matrix
