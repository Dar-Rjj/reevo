import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """Alpha factor combining multi-timeframe momentum, dynamic volume percentiles, and price efficiency with adaptive scaling."""
    # Multi-timeframe momentum: weighted 3-day and 10-day returns adjusted by exponential decay
    mom_short = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_medium = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    decay_factor = np.exp(-0.1 * np.arange(10))  # exponential decay for responsiveness
    composite_momentum = 0.4 * mom_short + 0.6 * mom_medium * decay_factor[-1]

    # Dynamic volume percentiles: regime-aware confirmation
    volume_percentile = df['volume'].rolling(20).apply(lambda x: np.percentile(x, 70))
    volume_adjustment = np.where(df['volume'] > volume_percentile, 1.2, 0.8)  # adaptive scaling
    volume_signal = np.log1p(df['volume'] / df['volume'].rolling(30).mean()) * volume_adjustment

    # Price efficiency: bounded sigmoid transformation for stability
    efficiency_raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    price_efficiency = 1 / (1 + np.exp(-10*(efficiency_raw - 0.5)))  # sharper sigmoid

    # Combine signals with multiplicative scaling preserving economic intuition
    alpha = composite_momentum * volume_signal * price_efficiency
    return alpha
