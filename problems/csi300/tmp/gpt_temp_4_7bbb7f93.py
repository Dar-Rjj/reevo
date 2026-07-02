import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor incorporating:
    1. Dynamic momentum/reversal blending using volatility regimes
    2. Cross-sectional ranking for comparability
    3. Volume divergence with price signals for regime confirmation
    4. Non-linear transforms (tanh/sqrt) for robustness
    5. Adaptive weights balancing short/medium/long-term signals
    6. Clear economic intuition and interpretability
    """

    # 1. Volatility-adjusted momentum blending
    true_range = np.maximum(df['high'] - df['low'],
                           np.maximum(abs(df['high'] - df['close'].shift()),
                                     abs(df['low'] - df['close'].shift())))
    atr = true_range.rolling(14).mean()
    vol_regime = np.tanh(atr / df['close'].rolling(20).std())

    # 2. Multi-timeframe momentum/reversal blending
    mom_short = df['close'].pct_change(5)
    mom_medium = -df['close'].pct_change(10)  # Reversal signal
    mom_long = df['close'].pct_change(20)

    momentum_signal = (
        np.tanh(mom_short * (1 + vol_regime)) * 0.4 +  # Momentum in low vol, neutral in high vol
        np.tanh(mom_medium * (1 - vol_regime)) * 0.3 +  # Reversal in high vol
        np.tanh(mom_long) * 0.3
    )

    # 3. Volume-ATR divergence with price confirmation
    volume_zscore = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-7)
    price_change_zscore = df['close'].pct_change().rolling(20).apply(lambda x: (x[-1] - x.mean()) / (x.std() + 1e-7))
    vol_price_divergence = np.sign(volume_zscore) * np.sign(price_change_zscore) * np.sqrt(abs(volume_zscore * price_change_zscore))

    # 4. Price efficiency ratio with non-linear transform
    range_utilization = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    efficiency_signal = np.tanh(range_utilization.rolling(5).mean() * np.log1p(atr))

    # 5. Cross-sectional ranking (pure ranking, no normalization)
    def cross_section_rank(series):
        return series.groupby(level=0).rank(pct=True)

    momentum_rank = cross_section_rank(momentum_signal)
    divergence_rank = cross_section_rank(vol_price_divergence)
    efficiency_rank = cross_section_rank(efficiency_signal)

    # 6. Adaptive blending with clear economic intuition
    alpha = (
        0.5 * momentum_rank +  # Primary driver: momentum/reversal blend
        0.3 * np.sqrt(abs(divergence_rank)) * np.sign(divergence_rank) +  # Emphasize directionality
        0.2 * efficiency_rank * momentum_rank  # Interaction term for efficiency confirmation
    )

    return alpha
