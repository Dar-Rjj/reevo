import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 1. Adaptive momentum blending with volatility-based weights
    mom_3d = df['close'].pct_change(3)
    mom_7d = df['close'].pct_change(7)
    mom_21d = df['close'].pct_change(21)
    reversal = -df['close'].pct_change(1)
    
    # Dynamic weights based on recent volatility
    recent_vol = df['close'].pct_change().rolling(5).std()
    vol_weight = np.tanh(recent_vol * 10)  # Squash to [0,1] range
    momentum_blend = (0.4 - 0.2*vol_weight)*mom_3d + (0.35 + 0.15*vol_weight)*mom_7d + (0.25 + 0.05*vol_weight)*mom_21d + (0.2*reversal)
    
    # 2. Liquidity-adjusted volatility scaling
    vol_10d = df['close'].pct_change().rolling(10).std()
    liquidity = np.log1p(df['volume'].rolling(5).mean())
    momentum_scaled = momentum_blend / (vol_10d + 1e-7) * (1 + 0.6*np.tanh(liquidity - 5))
    
    # 3. Smart intraday confirmation with price rejection
    gap = np.log(df['open'] / df['close'].shift(1))
    price_rejection = (2*df['close'] - df['low'] - df['high']) / (df['high'] - df['low'] + 1e-7)
    intraday_signal = gap * price_rejection * np.sign(gap)
    
    # 4. Non-linear combination with robustness
    raw_factor = (
        0.6 * momentum_scaled
        + 0.2 * np.tanh(intraday_signal * 3)
        + 0.2 * np.sign(momentum_scaled) * np.sqrt(np.abs(momentum_scaled))
    )
    
    # 5. Enhanced smoothing with asymmetric rolling median
    factor_vol = raw_factor.rolling(21).std()
    smoothed_factor = raw_factor.rolling(5, min_periods=1).apply(lambda x: np.median(x[x > 0]) if any(x > 0) else np.median(x[x < 0]) if any(x < 0) else 0)
    
    return smoothed_factor / (factor_vol + 1e-7).rolling(5).median()
