import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # 1. Ultra-short momentum (1-day % change)
    momentum = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # 2. Log-transformed volume ratio (current vs 3-day median)
    vol_ratio = df['volume'] / (df['volume'].rolling(3).median() + 1e-7)
    log_vol = np.sign(vol_ratio) * np.log1p(np.abs(vol_ratio))
    
    # 3. Volatility-normalized price range
    daily_range = (df['high'] - df['low']) / (df['close'].rolling(5).std() + 1e-7)
    
    # 4. Intraday strength with position weighting
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    range_weight = 2 * (close_position - 0.5)  # [-1,1] centered at 0.5
    
    # 5. Multiplicative combination
    factor = momentum * log_vol * daily_range * range_weight
    
    # 6. Hamming window smoothing (3-day)
    window = np.hamming(3)
    smoothed = factor.rolling(3, center=True).apply(lambda x: np.sum(x * window))
    
    # 7. Cross-sectional robustness (rank normalization)
    ranked = smoothed.rank(pct=True)
    
    return ranked
