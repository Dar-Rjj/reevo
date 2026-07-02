import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """Alpha factor combining refined momentum, dynamic efficiency, and enhanced volume regime."""
    # Refined momentum: shorter-term (5-day) and medium-term (14-day) blend
    momentum_short = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_medium = (df['close'] - df['close'].shift(14)) / df['close'].shift(14)
    true_range = (df['high'].rolling(5).max() - df['low'].rolling(5).min())
    momentum_blend = (0.4*momentum_short + 0.6*momentum_medium) / (true_range + 1e-7)

    # Dynamic efficiency: logistic scaling with adjusted sensitivity
    efficiency_raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    efficiency_scaled = (1 / (1 + np.exp(-12*(efficiency_raw-0.6)))) - 0.5

    # Enhanced volume regime: log-transformed, scaled by percentile, and sigmoid-bounded
    volume_percentile = df['volume'].rolling(20).apply(lambda x: abs(x).rank(pct=True).iloc[-1])
    volume_scaled = np.log1p(df['volume'] / (df['volume'].rolling(20).quantile(0.75) + 1e-7))
    volume_regime = 2 * (1 / (1 + np.exp(-6 * volume_scaled)) - 0.5)

    # Multiplicative combination for clear economic interpretation
    alpha = momentum_blend * efficiency_scaled * volume_regime
    return alpha
