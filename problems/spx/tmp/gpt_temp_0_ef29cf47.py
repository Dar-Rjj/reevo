def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Momentum (5-day rolling window rate of change)
    momentum = df['close'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    # Normalize High-Low Range by Momentum
    # Add a small constant to avoid division by zero
    factor = high_low_range / (momentum + 1e-6)
    
    return factor
