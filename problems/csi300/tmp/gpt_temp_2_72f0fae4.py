import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Enhanced momentum-reversion hybrid alpha factor with:
    # Volatility-weighted smoothing, orthogonal components, and interpretable transformations

    # 1. Momentum (15-day return, log-transformed with volatility adjustment)
    momentum = df['close'] / df['close'].shift(15) - 1
    volatility = df['close'].rolling(15).std() / df['close'].rolling(15).mean()
    mom15 = np.sign(momentum) * np.log1p(abs(momentum)) / (volatility + 1e-7)

    # 2. Reversal (5-day return, cube-root transformed with volume confirmation)
    reversal = -(df['close'] / df['close'].shift(5) - 1)
    rev5 = np.cbrt(reversal) * np.cbrt(df['volume'] / df['volume'].shift(5))

    # 3. Overnight gap (log-transformed, adjusted for market context)
    gap = np.log1p(df['close'].shift(1) / df['open'] - 1)

    # 4. Volume trend (log-transformed 10-day volume slope)
    vol_slope = df['volume'].rolling(10).apply(lambda x: np.polyfit(range(10), x, 1)[0])
    vol_trend = np.log1p(vol_slope)

    # 5. Price efficiency (log-transformed close-to-close vs high-low range)
    range_eff = np.log1p((df['close'] - df['close'].shift(1)).abs() / (df['high'] - df['low'] + 1e-7))

    # 6. Market trend context (20-day exponential moving average slope, sigmoid-weighted)
    ema_slope = df['close'].ewm(span=20).mean().diff()
    trend_weight = 1 / (1 + np.exp(-ema_slope * 0.5))

    # Composite factor:
    # Momentum favored in bullish trends, reversal in bearish trends
    # Volume-adjusted signals, orthogonal components, and volatility-scaled smoothing
    factor = trend_weight * mom15 * range_eff * vol_trend + (1 - trend_weight) * (rev5 + gap) * (1 - range_eff)

    return factor
