import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """Alpha factor combining multi-timeframe momentum, dynamic volume confirmation, and price efficiency with adaptive scaling."""
    # Multi-timeframe momentum: 3-day and 10-day returns with exponential decay
    mom_short = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_medium = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    decay_factor = 0.8  # Exponential decay factor for responsiveness
    composite_momentum = (0.3 * mom_short + 0.7 * mom_medium) * decay_factor

    # Dynamic volume confirmation: percentile-based scaling
    volume_percentile = df['volume'].rolling(30).apply(lambda x: x.rank(pct=True).iloc[-1])
    volume_signal = np.tanh(volume_percentile * 2)  # Adaptive scaling for stronger signals

    # Price efficiency: range efficiency normalized by volatility
    range_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    volatility = (df['high'] - df['low']).rolling(5).mean()
    price_efficiency = range_efficiency / (volatility + 1e-7)

    # Multiplicative combination preserves economic intuition and adapts to market regimes
    alpha = composite_momentum * volume_signal * price_efficiency
    return alpha
