def heuristics_v2(df):
    # Calculate Intraday Momentum
    intraday_range = df['high'] - df['low']
    normalized_momentum = (intraday_range / df['close']) - 1
    
    # Confirm with Volume Trend
    volume_surge = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0])
    factor = normalized_momentum * volume_surge
    
    # Normalize by Mean Volume
    mean_volume = df['volume'].rolling(window=5).mean()
    normalized_factor = factor / mean_volume
    
    return normalized_factor
