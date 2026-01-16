def heuristics_v2(df):
    # Calculate Open-to-Close Momentum
    momentum = df['close'] - df['open']
    
    # Calculate High-Low Range
    price_range = df['high'] - df['low']
    
    # Normalize Momentum by Price Range
    normalized_momentum = momentum / price_range
    
    # Compute Volume Slope over 5 days
    volume_slope = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / len(x))
    
    # Validate with Volume Trend
    factor = normalized_momentum * volume_slope
    
    return factor
