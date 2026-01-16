def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate 5-day rolling momentum of High-Low Range
    momentum = high_low_range.rolling(5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    return momentum
