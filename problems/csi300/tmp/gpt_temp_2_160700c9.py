def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Normalize High-Low Range by Close price
    df['normalized_range'] = df['high_low_range'] / df['close']
    
    # Calculate Rolling Market Average High-Low Range
    rolling_sum_range = df.groupby('date')['high_low_range'].transform(lambda x: x.expanding().sum())
    rolling_count = df.groupby('date').cumcount() + 1
    df['rolling_market_avg_range'] = rolling_sum_range / rolling_count
    
    # Subtract Rolling Market Average from Normalized Range
    factor = df['normalized_range'] - df['rolling_market_avg_range']
    
    return factor.dropna()
