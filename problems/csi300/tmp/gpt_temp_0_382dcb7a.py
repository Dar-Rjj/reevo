import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """Enhanced momentum-volume-range factor with decay weights and bounded transforms."""
    # Momentum component: blended horizons with exponential decay weights
    decay_short = 0.8 ** np.arange(5)[::-1]  # 5-day decay
    decay_long = 0.95 ** np.arange(20)[::-1]  # 20-day decay
    weighted_short = (df['close'].rolling(5).apply(lambda x: np.sum(x * decay_short)) / np.sum(decay_short))
    weighted_long = (df['close'].rolling(20).apply(lambda x: np.sum(x * decay_long)) / np.sum(decay_long))
    momentum = (weighted_short - weighted_long) / (df['close'].shift(1) + 1e-7)

    # Volatility-scaled true range adjustment
    atr = (df['high'] - df['low']).rolling(10).mean()
    scaled_momentum = momentum / (atr + 1e-7)

    # Volume confirmation using logistic percentiles
    vol_rank = df['volume'].rolling(50).rank(pct=True)
    vol_conf = 2 / (1 + np.exp(-5*(vol_rank-0.6))) - 1  # S-curve around 60th percentile

    # Bounded range efficiency (tanh transform)
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    range_eff = np.tanh(3*(daily_efficiency - 0.5))  # [-1,1] bounds

    # Multiplicative combination with economic interpretation:
    # Momentum × Volume Confirmation × Range Efficiency
    alpha = scaled_momentum * vol_conf * range_eff
    return alpha
