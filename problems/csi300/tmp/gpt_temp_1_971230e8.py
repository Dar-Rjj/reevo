import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-horizon momentum with adaptive weighting
    momentum_3d = np.log(df['close']) - np.log(df['close'].shift(3))
    momentum_10d = np.log(df['close']) - np.log(df['close'].shift(10))
    momentum_30d = np.log(df['close']) - np.log(df['close'].shift(30))
    
    # Dynamic weighting based on recent performance
    recent_perf = momentum_3d.rolling(10).std() / (momentum_30d.rolling(10).std() + 1e-7)
    weighted_momentum = (momentum_3d * recent_perf + momentum_10d + momentum_30d / recent_perf) / (recent_perf + 1 + 1/recent_perf)
    
    # Robust volatility adjustment using median absolute deviation
    mad = (df['close'].pct_change().abs().rolling(30).median() * 1.4826 + 1e-7)
    volatility_adjusted = weighted_momentum / mad
    
    # Volume-price efficiency with log transforms
    log_volume = np.log1p(df['volume'])
    log_range = np.log1p(df['high'] - df['low'])
    volume_efficiency = log_volume.diff(5) / (log_range.rolling(5).mean() + 1e-7)
    
    # Orthogonal combination with interaction term
    combined = volatility_adjusted * volume_efficiency * np.sign(volatility_adjusted * volume_efficiency)
    
    # Two-stage robust smoothing
    smoothed = combined.rolling(window=10, min_periods=5).apply(lambda x: np.nanmedian(x[x > np.percentile(x, 25)]))
    smoothed = smoothed.rolling(window=5, min_periods=3).mean()
    
    # Cross-sectional rank with outlier trimming
    def trimmed_rank(s):
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        iqr = q75 - q25
        return s.clip(lower=q25-1.5*iqr, upper=q75+1.5*iqr).rank(pct=True)
    
    factor_ranked = smoothed.groupby(level=0).apply(trimmed_rank)
    
    return factor_ranked
