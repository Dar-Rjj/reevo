def heuristics_v2(df):
    # High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Normalize by Close Price
    normalized_range = high_low_range / df['close']
    
    # Rolling Price Momentum
    rolling_window = 10
    rolling_median = df['close'].rolling(window=rolling_window, min_periods=1).median()
    
    # Compute Deviation
    rolling_mad = df['close'].rolling(window=rolling_window, min_periods=1).apply(lambda x: (x - x.median()).abs().mean())
    deviation = (df['close'] - rolling_median) / rolling_mad
    
    # Factor Calculation
    factor = normalized_range * deviation
    
    return factor
