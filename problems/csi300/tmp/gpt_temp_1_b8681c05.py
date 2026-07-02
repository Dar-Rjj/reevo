import pandas as pd

def heuristics_v2(df):
    {"Combines price volatility, volume-weighted price acceleration, and directional price change with a rolling geometric mean."}
    price_volatility = (df['high'] - df['low']) / df['close']
    volume_weighted_acceleration = df['volume'] * (df['close'].diff(2) / df['close'].shift(2))
    directional_change = df['close'].pct_change(5)
    heuristics_matrix = (price_volatility * volume_weighted_acceleration * directional_change).rolling(10).apply(lambda x: x.prod() ** (1/3))
    return heuristics_matrix
