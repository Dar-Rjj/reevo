import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Primitive operations
    def sma(series, window=5):
        return series.rolling(window=window).mean()
    
    def ema(series, window=5):
        return series.ewm(span=window, adjust=False).mean()
    
    def roc(series, window=1):
        return series.pct_change(window)
    
    def stddev(series, window=5):
        return series.rolling(window=window).std()
    
    def corr(x, y, window=5):
        return x.rolling(window=window).corr(y)
    
    def ts_rank(series, window=5):
        return series.rolling(window=window).apply(lambda x: pd.Series(x).rank().iloc[-1] / window)
    
    # Feature combinations
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    high = df['high']
    low = df['low']
    open_ = df['open']
    
    # Heuristic factors
    factor1 = -roc(close, 1) * ts_rank(volume, 5)
    factor2 = corr(high, low, 5) * (high + low) / 2
    factor3 = sma(amount, 3) / ema(volume, 5)
    factor4 = (close - sma(close, 10)) / stddev(close, 10)
    factor5 = (high / low) * roc(open_, 2)
    
    # Combine factors
    heuristics_matrix = factor1 + factor2 + factor3 + factor4 + factor5
    heuristics_matrix.name = 'heuristic_factor'
    
    return heuristics_matrix
