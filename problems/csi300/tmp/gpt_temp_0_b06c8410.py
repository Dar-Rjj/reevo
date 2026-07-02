import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Enhanced momentum-reversion hybrid with:
    # Longer-term momentum (90D), dynamic regime detection, volume confirmation, and orthogonal components
    
    # 1. Long-term momentum (90-day log return, smoothed)
    mom90 = np.log(df['close'] / df['close'].shift(90))
    mom90_smoothed = mom90.ewm(span=20).mean()
    
    # 2. Short-term reversal (3-day cube root return)
    rev3 = -np.cbrt(df['close'] / df['close'].shift(3) - 1)
    
    # 3. Overnight gap persistence (log gap * sign consistency)
    gap = np.log1p(df['close'].shift(1) / df['open'] - 1)
    gap_sign_consistency = gap.rolling(5).apply(lambda x: np.mean(np.sign(x) == np.sign(x[-1])))
    
    # 4. Volume confirmation (log volume / 60D median)
    vol_ratio = np.log1p(df['volume'] / (df['volume'].rolling(60).median() + 1e-7))
    
    # 5. Range efficiency (close-to-close vs true range)
    true_range = np.maximum.reduce([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ])
    range_eff = (df['close'] - df['close'].shift(1)).abs() / (true_range + 1e-7)
    
    # 6. Market regime detection (60D volatility-adjusted EMA slope)
    ema_slope = df['close'].ewm(span=60).mean().diff()
    vol_60d = df['close'].pct_change().rolling(60).std()
    regime_strength = ema_slope / (vol_60d + 1e-7)
    
    # 7. Dynamic weighting (sigmoid transition between regimes)
    momentum_weight = 1 / (1 + np.exp(-regime_strength * 5))
    
    # Composite factor:
    # - Momentum regime: long momentum + range efficiency
    # - Reversal regime: short reversal + overnight gaps
    # Both regimes volume-adjusted
    factor = (
        momentum_weight * mom90_smoothed * np.sqrt(range_eff) * vol_ratio +
        (1 - momentum_weight) * (rev3 + gap * gap_sign_consistency) * (1 - range_eff) * vol_ratio
    )
    
    return factor
