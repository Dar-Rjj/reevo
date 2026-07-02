import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-horizon momentum with weighted blending
    momentum_3d = df['close'] / df['close'].rolling(window=3, min_periods=2).median()
    momentum_10d = df['close'] / df['close'].rolling(window=10, min_periods=5).median()
    momentum_30d = df['close'] / df['close'].rolling(window=30, min_periods=10).median()
    blended_momentum = 0.4*momentum_3d + 0.4*momentum_10d + 0.2*momentum_30d
    
    # Robust volatility scaling using median absolute deviation
    price_range = df['high'] - df['low']
    volatility = price_range.rolling(window=10, min_periods=5).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    risk_adjusted_momentum = blended_momentum / (volatility + 1e-7)
    
    # Volume efficiency with robust scaling
    log_volume = np.log1p(df['volume'])
    log_range = np.log1p(price_range)
    volume_efficiency = log_volume / (log_range.rolling(window=5, min_periods=3).median() + 1e-7)
    
    # VWAP divergence with robust normalization
    vwap = (df['high'] + df['low'] + df['close']) / 3
    vwap_divergence = (df['close'] - vwap) / (
        vwap.rolling(window=10, min_periods=5).std() + 1e-7
    )
    
    # Orthogonal factor combination
    combined_factor = risk_adjusted_momentum * volume_efficiency * vwap_divergence
    
    # Robust smoothing with median filter
    smoothed_factor = combined_factor.rolling(window=5, min_periods=3).median()
    
    # Cross-sectional ranking with outlier trimming
    ranked_factor = smoothed_factor.groupby(level=0).apply(
        lambda x: x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95)).rank(pct=True)
    )
    
    return ranked_factor
