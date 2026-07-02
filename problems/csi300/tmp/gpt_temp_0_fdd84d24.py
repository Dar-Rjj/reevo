def heuristics_v2(df):
    # Momentum Breakout Component
    normalized_range_momentum = (df['high'] - df['low']) / df['close']
    rolling_mean_momentum = normalized_range_momentum.rolling(window=5, min_periods=1).mean()
    momentum_signal = normalized_range_momentum - rolling_mean_momentum
    
    # Breakout Signal
    breakout_signal = momentum_signal.apply(lambda x: 1 if x > 1.2 * momentum_signal.rolling(window=5, min_periods=1).mean().iloc[-1] else 0)
    
    # Volume Confirmation Component
    volume_ratio = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    volume_signal = volume_ratio.apply(lambda x: 1 if x > 1.5 else 0)
    
    # Combined Signal
    combined_signal = breakout_signal * volume_signal
    combined_signal = combined_signal.clip(-1, 1)
    
    return combined_signal
