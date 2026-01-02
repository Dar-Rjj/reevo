import numpy as np
def heuristics_v2(df):
    # Compute Raw Intraday Momentum
    high_to_close = (df['high'] - df['close']) / df['close']
    low_to_close = (df['low'] - df['close']) / df['close']
    raw_momentum = high_to_close + low_to_close
    
    # Volatility-Adjusted Signal
    rolling_std = df['close'].rolling(window=20, min_periods=1).std()
    volatility_adjusted = raw_momentum / (rolling_std + 1e-6)  # Add small constant to avoid division by zero
    
    # Volume-Weighted Smoothing
    volume_weighted_signal = (
        df['volume'].rolling(window=5, min_periods=1).apply(lambda x: (x * volatility_adjusted.loc[x.index]).sum(), raw=False) / 
        df['volume'].rolling(window=5, min_periods=1).sum()
    )
    smoothed_signal_diff = volume_weighted_signal - volume_weighted_signal.shift(5)
    
    # Directional Confirmation
    price_trend = df['close'].diff(5).apply(np.sign)
    signal_trend = smoothed_signal_diff.apply(np.sign)
    trend_alignment = (price_trend == signal_trend).astype(int)
    
    # Filter Weak Signals
    signal_threshold = smoothed_signal_diff.abs().quantile(0.25)
    filtered_signal = smoothed_signal_diff.where(
        (smoothed_signal_diff.abs() > signal_threshold) & (trend_alignment == 1),
        0
    )
    
    return filtered_signal
