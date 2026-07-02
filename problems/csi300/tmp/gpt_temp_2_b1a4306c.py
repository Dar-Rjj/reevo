import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Price momentum with adaptive lookback (5-10 day crossover)
    short_ma = df['close'].ewm(span=5).mean()
    long_ma = df['close'].ewm(span=10).mean()
    momentum = (short_ma - long_ma) / df['close'].rolling(window=10).std()
    
    # Volume acceleration (current vs historical trend)
    volume_trend = df['volume'].rolling(window=10).apply(lambda x: np.polyfit(range(10), x, 1)[0])
    volume_accel = df['volume'] / (df['volume'].rolling(window=10).mean() + volume_trend)
    
    # Normalized volatility (range relative to recent volatility)
    daily_range = (df['high'] - df['low']) / df['close']
    vol_adjusted_range = daily_range / daily_range.rolling(window=10).std()
    
    # Price-volume correlation (3-day rolling)
    pv_corr = df['close'].rolling(window=3).corr(df['volume'])
    
    # Combined factor with interaction terms
    alpha_factor = (momentum * volume_accel * vol_adjusted_range) * (1 + pv_corr)
    
    return alpha_factor
