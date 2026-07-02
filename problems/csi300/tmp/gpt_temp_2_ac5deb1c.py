import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """Alpha factor blending medium-term momentum, adaptive volatility adjustment, and dynamic volume confirmation."""
    # Medium-term momentum (30D) with exponential decay weights for robustness
    weights = np.exp(-np.arange(30) * 0.1)  # Exponential decay weights
    weights /= weights.sum()  # Normalize weights to sum to 1
    momentum = df['close'].diff(30).rolling(30).apply(lambda x: (x * weights[::-1]).sum(), raw=True) / df['close'].shift(30)

    # Adaptive volatility adjustment using modified ATR (10-day window)
    atr = (df['high'] - df['low']).rolling(10).mean()
    adjusted_momentum = momentum / (atr + 1e-7)

    # Smooth dynamic volume confirmation using log-transformed volume percentiles
    volume_percentiles = df['volume'].rolling(50).apply(lambda x: np.percentile(x, 50)) / df['volume']
    volume_confirmation = np.tanh(np.log1p(volume_percentiles))

    # Multiplicative combination for interpretable interaction effects
    alpha = adjusted_momentum * volume_confirmation
    return alpha
