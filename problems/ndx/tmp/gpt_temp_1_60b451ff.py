import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 1. Multi-scale momentum with dynamic blending
    mom_3d = df['close'].pct_change(3)
    mom_7d = df['close'].pct_change(7)
    mom_21d = df['close'].pct_change(21)
    reversal = -df['close'].pct_change(1)  # Short-term mean reversion
    
    # Volatility-adaptive momentum blending
    recent_vol = df['close'].pct_change().rolling(5).std()
    vol_weight = np.tanh(recent_vol * 10)  # Adaptive to volatility spikes
    momentum_blend = (
        (0.5 - 0.3*vol_weight)*mom_3d 
        + (0.35 + 0.15*vol_weight)*mom_7d 
        + (0.15 + 0.15*vol_weight)*mom_21d 
        + (0.2*reversal)*(1 - vol_weight)  # Stronger reversal in low vol
    )
    
    # 2. Liquidity-aware signal scaling
    vol_10d = df['close'].pct_change().rolling(10).std()
    liquidity = np.log1p(df['volume'].rolling(5).mean())
    momentum_scaled = momentum_blend / (vol_10d + 1e-7) * np.sqrt(1 + liquidity/7)
    
    # 3. Intraday confirmation with gap and rejection patterns
    gap = np.log(df['open'] / df['close'].shift(1))
    rejection = (2*df['close'] - df['low'] - df['high']) / (df['high'] - df['low'] + 1e-7)
    intraday_confirmation = gap * rejection * np.power(np.abs(gap), 0.75) * np.sign(gap + rejection)
    
    # 4. Non-linear combination with momentum anchoring
    raw_factor = (
        0.6 * momentum_scaled 
        + 0.25 * np.tanh(intraday_confirmation * 2.5)
        + 0.15 * np.sign(momentum_scaled) * np.power(np.abs(momentum_scaled), 0.7)
    )
    
    # 5. Volatility-normalized smoothing with decay
    factor_vol = raw_factor.rolling(21).std()
    decay_weight = np.exp(-recent_vol * 5)  # Faster decay in high volatility
    smoothed_factor = raw_factor.rolling(3, min_periods=1).apply(
        lambda x: np.nanmean(x * decay_weight)
    )
    return smoothed_factor / (factor_vol + 1e-7).rolling(5).median()
