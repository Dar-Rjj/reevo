import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Core components (multiplicative interactions)
    vwap = (df['high'] + df['low'] + df['close']) / 3
    
    # 1. Volatility-anchored momentum (20D vs 5D)
    momentum = (df['close'] / df['close'].rolling(window=20).median()) ** 2
    vol_adj = np.log1p((df['high'] - df['low']).rolling(window=10).std())
    momentum_vol_scaled = momentum / (vol_adj + 1e-7)
    
    # 2. Volume efficiency: log(volume)/range
    vol_eff = np.log1p(df['volume']) / (df['high'] - df['low'] + 1e-7)
    vol_eff_smoothed = vol_eff.rolling(window=15, min_periods=5).median()
    
    # 3. VWAP convergence (multi-horizon)
    vwap_conv = (vwap / df['close'].rolling(window=5).mean()) * \
                (vwap / df['close'].rolling(window=10).mean()) ** 0.5
    
    # Combined factor (multiplicative interaction)
    raw_alpha = momentum_vol_scaled * vol_eff_smoothed * vwap_conv
    
    # Robust processing
    alpha_processed = raw_alpha.rolling(window=10).apply(
        lambda x: np.nanmedian(x) if len(x.dropna()) >= 5 else np.nan
    )
    
    # Cross-sectional rank with clipping
    alpha_ranked = alpha_processed.groupby(level=0).apply(
        lambda x: x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95)).rank(pct=True)
    )
    
    return alpha_ranked
