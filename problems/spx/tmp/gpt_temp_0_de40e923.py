def heuristics_v2(df):
    # Calculate Rolling Price Momentum
    df['close_slope'] = df['close'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / (x[0] + 1e-9))
    
    # Calculate Rolling Mean Volume
    df['mean_volume'] = df['volume'].rolling(window=5).mean()
    
    # Filter by Rolling Volume
    factor = df['close_slope'].where(df['volume'] > df['mean_volume'])
    
    return factor
