import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Triple-horizon momentum blend (1d, 5d, 20d) with decay weights
    momentum_1d = df['close'] / df['close'].shift(1)
    momentum_5d = df['close'] / df['close'].shift(5)
    momentum_20d = df['close'] / df['close'].shift(20)
    blended_momentum = (0.5 * momentum_1d + 0.3 * momentum_5d + 0.2 * momentum_20d)
    
    # Adaptive volatility scaling using rolling percentiles (20d window)
    true_range = df['high'] - df['low']
    vol_measure = true_range.rolling(window=20).apply(lambda x: np.percentile(x, 75))
    scaled_momentum = blended_momentum / (vol_measure + 1e-7)
    
    # Volume confirmation with asymmetric decay (faster reaction to increases)
    volume_ma = df['volume'].ewm(span=10, adjust=False).mean()
    volume_confirmation = np.where(
        df['volume'] > volume_ma,
        np.sqrt(df['volume'] / volume_ma),
        (df['volume'] / volume_ma) ** 2
    )
    
    # Price efficiency measure using intraday capture
    open_to_close = df['close'] / df['open']
    intraday_range = df['high'] / df['low']
    price_efficiency = open_to_close / intraday_range
    
    # Combine orthogonal signals with non-linear interaction
    composite = scaled_momentum * np.log1p(volume_confirmation) * price_efficiency
    
    # Robust smoothing with double median filters
    temp_smooth = composite.rolling(window=3).median()
    final_smooth = temp_smooth.rolling(window=5).median()
    
    # Cross-sectional rank transformation (preserves distribution)
    factor = final_smooth.groupby(level=0).rank(pct=True)
    
    return factor
