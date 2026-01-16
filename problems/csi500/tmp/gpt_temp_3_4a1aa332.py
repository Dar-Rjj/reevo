def heuristics_v2(df):
    # Calculate Intraday Momentum
    price_momentum = (df['close'] - df['open']) / df['open']
    volume_ma = df['volume'].rolling(20).mean()
    volume_spike = df['volume'] / volume_ma
    momentum_signal = price_momentum * volume_spike
    
    # Measure Intraday Price Range
    price_range = (df['high'] - df['low']) / df['open']
    returns = df['close'].pct_change()
    historical_vol = returns.rolling(20).std()
    normalized_range = price_range / historical_vol
    
    # Combine Signals
    combined_signal = momentum_signal * normalized_range
    zscore_signal = combined_signal.rolling(10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Detect Breakout and Adjust Signal
    rolling_high = df['high'].rolling(5).max()
    rolling_low = df['low'].rolling(5).min()
    
    upper_breakout = df['close'] > rolling_high
    lower_breakout = df['close'] < rolling_low
    
    final_signal = zscore_signal.copy()
    final_signal[upper_breakout] = zscore_signal[upper_breakout] * 1
    final_signal[lower_breakout] = zscore_signal[lower_breakout] * -1
    
    return final_signal
