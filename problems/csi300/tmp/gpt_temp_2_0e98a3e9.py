import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced momentum-reversion hybrid factor with:
    - Multiple time horizon integration
    - Robust non-linear transforms
    - Trend-adaptive regime switching
    - Intraday price efficiency signals
    """
    
    # 1. Multi-horizon momentum components (cube-root transformed)
    mom5 = np.cbrt(df['close'] / df['close'].shift(5) - 1)  # Short-term
    mom20 = np.cbrt(df['close'] / df['close'].shift(20) - 1)  # Medium-term
    
    # 2. Multi-horizon reversal components (cube-root transformed)
    rev1 = -np.cbrt(df['close'] / df['close'].shift(1) - 1)  # Ultra-short
    rev3 = -np.cbrt(df['close'] / df['close'].shift(3) - 1)  # Short-term
    
    # 3. Overnight gap component (cube-root transformed)
    gap = np.cbrt(df['open'] / df['close'].shift(1) - 1)
    
    # 4. Volume confirmation (cube-root transformed volume / 20-day median)
    vol_ratio = np.cbrt(df['volume'] / (df['volume'].rolling(20).median() + 1e-7))
    
    # 5. Price efficiency metrics
    range_eff = np.cbrt((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-7))  # Intraday
    close_eff = np.cbrt((df['close'] - df['close'].shift(1)).abs() / (df['high'] - df['low'] + 1e-7))  # Interday
    
    # 6. Trend context (dual moving average system)
    fast_ma = df['close'].ewm(span=20).mean()
    slow_ma = df['close'].ewm(span=50).mean()
    trend_strength = np.tanh((fast_ma - slow_ma) * 3)  # Smoothed trend indicator
    
    # 7. Volatility regime (robust median deviation)
    ret_vol = df['close'].pct_change().rolling(20).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    vol_regime = np.tanh(ret_vol * 10)  # Scaled volatility regime
    
    # Composite factor construction:
    # - Momentum dominates in strong trends with low volatility
    # - Reversal dominates in weak trends with high volatility
    # - Volume and efficiency modulate signal strength
    factor = (
        (0.5 * trend_strength + 0.5) * (0.6*mom20 + 0.4*mom5) * (0.7*range_eff + 0.3*close_eff) * vol_ratio * (1 - vol_regime) +
        (0.5 - 0.5 * trend_strength) * (0.5*rev3 + 0.3*rev1 + 0.2*gap) * (1.5 - (0.7*range_eff + 0.3*close_eff)) * vol_ratio * vol_regime
    )
    
    return factor
