import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # High-Low Volatility Skewness
    daily_range = data['high'] - data['low']
    range_20d = daily_range.rolling(window=20, min_periods=10).mean()
    range_skewness = daily_range.rolling(window=5, min_periods=3).apply(
        lambda x: x.skew() if len(x) >= 3 else np.nan
    )
    factor1 = range_skewness / (range_20d + 1e-8)
    
    # Volume-Adjusted Price Momentum Divergence
    price_momentum = data['close'].pct_change(periods=10)
    volume_change = data['volume'].pct_change(periods=10)
    factor2 = price_momentum - volume_change
    
    # Opening Gap Persistence
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_direction = np.sign(opening_gap)
    gap_persistence = gap_direction.rolling(window=3, min_periods=2).apply(
        lambda x: np.mean(x == x.iloc[-1]) if len(x) >= 2 else np.nan
    )
    factor3 = gap_persistence * np.abs(opening_gap)
    
    # Intraday Reversal Strength
    intraday_range = data['high'] - data['low']
    close_to_open = np.abs(data['close'] - data['open'])
    reversal_magnitude = (intraday_range - close_to_open) / (intraday_range + 1e-8)
    volume_ratio = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    factor4 = reversal_magnitude * volume_ratio
    
    # Price-Volume Acceleration Divergence
    price_returns = data['close'].pct_change()
    price_accel = price_returns.diff().rolling(window=5, min_periods=3).mean()
    volume_returns = data['volume'].pct_change()
    volume_accel = volume_returns.diff().rolling(window=5, min_periods=3).mean()
    factor5 = price_accel - volume_accel
    
    # Resistance Breakout Confirmation
    resistance_level = data['high'].rolling(window=10, min_periods=5).max()
    breakout_signal = (data['close'] > resistance_level.shift(1)).astype(int)
    volume_surge = data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean()
    factor6 = breakout_signal * volume_surge * (data['close'] - resistance_level.shift(1)) / resistance_level.shift(1)
    
    # Low-Volume Trend Continuation
    volume_20d_avg = data['volume'].rolling(window=20, min_periods=10).mean()
    low_volume_flag = (data['volume'] < volume_20d_avg).astype(int)
    price_trend = data['close'].pct_change(periods=3)
    factor7 = low_volume_flag * price_trend
    
    # Amplitude-Volume Correlation Reversal
    daily_amplitude = (data['high'] - data['low']) / data['close']
    corr_window = 8
    amplitude_volume_corr = daily_amplitude.rolling(window=corr_window, min_periods=4).corr(data['volume'])
    corr_reversal = -amplitude_volume_corr.diff(periods=2)
    factor8 = corr_reversal
    
    # Close Position Strength
    close_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    volume_trend = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    factor9 = close_position * volume_trend
    
    # Multi-timeframe Momentum Alignment
    short_momentum = data['close'].pct_change(periods=3)
    medium_momentum = data['close'].pct_change(periods=8)
    momentum_alignment = np.sign(short_momentum) * np.sign(medium_momentum)
    momentum_strength = (np.abs(short_momentum) + np.abs(medium_momentum)) / 2
    factor10 = momentum_alignment * momentum_strength
    
    # Combine all factors with equal weights
    factors = [factor1, factor2, factor3, factor4, factor5, 
               factor6, factor7, factor8, factor9, factor10]
    
    # Standardize each factor and combine
    combined_factor = pd.Series(0, index=data.index)
    for f in factors:
        f_standardized = (f - f.rolling(window=20, min_periods=10).mean()) / (f.rolling(window=20, min_periods=10).std() + 1e-8)
        combined_factor = combined_factor + f_standardized
    
    return combined_factor
