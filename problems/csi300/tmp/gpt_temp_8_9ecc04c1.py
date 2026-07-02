import pandas as pd
import numpy as np

def heuristics_v2(df):
    {
        "Algorithm description": "Combines normalized price volatility, volume acceleration, and medium-term momentum with rolling geometric mean and adaptive smoothing."
    }
    price_volatility = (df['high'] - df['low']) / (df['close'].rolling(10).std() + 1e-6)
    volume_acceleration = df['volume'] / df['volume'].rolling(30).mean()
    momentum = df['close'].pct_change(5)
    heuristics_matrix = (price_volatility * volume_acceleration * momentum).rolling(15).apply(np.prod).ewm(span=20).mean()
    return heuristics_matrix
