import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Price acceleration (3-day vs 5-day momentum ratio)
    mom_3d = (df['close'] - df['close'].shift(3)) / (df['close'].shift(3) + 1e-7)
    mom_5d = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-7)
    price_accel = mom_3d / (mom_5d + np.sign(mom_5d)*1e-7)
    
    # Volume-adjusted price change (1-day return * log volume)
    log_volume = np.log(df['volume'] + 1)
    vol_adjusted_return = (df['close'].pct_change(1) + 1e-7) * log_volume
    
    # Range efficiency (absolute return / true range)
    true_range = df['high'].rolling(2).max() - df['low'].rolling(2).min()
    range_efficiency = df['close'].pct_change(1).abs() / (true_range + 1e-7)
    
    # Gap persistence (intraday move / overnight move)
    overnight_move = (df['open'] - df['close'].shift(1)).abs()
    intraday_move = (df['close'] - df['open']).abs()
    gap_persistence = intraday_move / (overnight_move + 1e-7)
    
    # Composite factor with complementary combinations
    factor = (
        price_accel * 
        np.sign(vol_adjusted_return) * np.sqrt(np.abs(vol_adjusted_return)) * 
        range_efficiency * 
        gap_persistence
    )
    
    # Robust smoothing with triangular window
    window_size = 7
    weights = np.bartlett(window_size)
    factor = factor.rolling(window=window_size, center=True).apply(
        lambda x: np.sum(x * weights / weights.sum())
    )
    
    return factor
