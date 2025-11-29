import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    prev_close = df['close'].shift(1)
    
    # Factor 1: Intraday Reversal with Volume Exhaustion
    morning_momentum = (df['high'] - df['open']) / prev_close
    afternoon_momentum = (df['close'] - df['low']) / prev_close
    
    # Estimate volume patterns (using rolling windows for hourly approximations)
    daily_volume = df['volume']
    first_hour_volume_est = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: x.iloc[0] if len(x) > 0 else np.nan, raw=False)
    last_hour_volume_est = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: x.iloc[-1] if len(x) > 0 else np.nan, raw=False)
    
    early_volume_concentration = first_hour_volume_est / daily_volume
    late_volume_drying = last_hour_volume_est / first_hour_volume_est
    
    volume_exhaustion = early_volume_concentration * late_volume_drying
    session_divergence = morning_momentum - afternoon_momentum
    factor1 = session_divergence * volume_exhaustion
    
    # Factor 2: Gap-Fill Acceleration with Range Contraction
    gap_size = np.abs(df['open'] - prev_close)
    gap_fill_progress = np.where(gap_size > 0, (df['close'] - df['open']) / gap_size, 0)
    
    # Estimate hourly fill rate using intraday range progression
    current_range = (df['high'] - df['low']) / prev_close
    range_3day_avg = current_range.rolling(window=3, min_periods=1).mean()
    range_contraction = current_range / range_3day_avg
    
    # Gap fill velocity estimation using intraday price movement
    fill_velocity = (df['close'] - df['open']) / (df['high'] - df['low'])
    fill_velocity = np.where((df['high'] - df['low']) > 0, fill_velocity, 0)
    
    factor2 = gap_fill_progress * range_contraction * np.sign(fill_velocity)
    
    # Factor 3: Volatility Breakout Position Reversal
    current_range_abs = df['high'] - df['low']
    range_10day_avg = current_range_abs.rolling(window=10, min_periods=1).mean()
    range_expansion = np.where(range_10day_avg > 0, current_range_abs / range_10day_avg, 1)
    
    opening_position_extreme = np.abs(df['open'] - prev_close) / prev_close
    hl_midpoint = (df['high'] + df['low']) / 2
    closing_reversal = np.where((df['high'] - df['low']) > 0, 
                               np.abs(df['close'] - hl_midpoint) / (df['high'] - df['low']), 0)
    
    volatility_breakout = (range_expansion > 1.2).astype(float)
    factor3 = opening_position_extreme * closing_reversal * volatility_breakout
    
    # Factor 4: Momentum-Volume Divergence with Session Asymmetry
    pre_noon_momentum = np.where(df['open'] > 0, (df['high'] - df['open']) / df['open'], 0)
    post_noon_momentum = np.where(df['low'] > 0, (df['close'] - df['low']) / df['low'], 0)
    
    # Estimate AM/PM volume split (using rolling patterns)
    am_volume_est = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: np.mean(x[:min(3, len(x))]) if len(x) > 0 else np.nan, raw=False)
    pm_volume_est = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: np.mean(x[max(0, len(x)-2):]) if len(x) > 0 else np.nan, raw=False)
    
    morning_volume_dominance = np.where(pm_volume_est > 0, am_volume_est / pm_volume_est, 1)
    
    # Volume-weighted session returns
    am_return_weighted = pre_noon_momentum * am_volume_est
    pm_return_weighted = post_noon_momentum * pm_volume_est
    volume_momentum_alignment = am_return_weighted - pm_return_weighted
    
    session_crossover = pre_noon_momentum - post_noon_momentum
    volume_skew = morning_volume_dominance - 1
    
    factor4 = session_crossover * volume_skew * volume_momentum_alignment
    
    # Combine factors with equal weighting
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + 
                      factor3.fillna(0) + factor4.fillna(0)) / 4
    
    result = combined_factor
    
    return result
