def heuristics_v2(df):
    # Calculate Normalized Range
    df['normalized_range'] = (df['high'] - df['low']) / df['open']
    
    # Calculate 5-Day Average Range
    df['5_day_avg_range'] = df['normalized_range'].rolling(window=5, min_periods=1).mean()
    
    # Compute Breakout Ratio
    df['breakout_ratio'] = df['normalized_range'] / df['5_day_avg_range']
    
    # Calculate Volume Surprise
    df['10_day_avg_volume'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_surprise'] = df['volume'] / df['10_day_avg_volume']
    
    # Create Liquidity Filter
    df['liquidity_flag'] = (df['volume_surprise'] > 1.25).astype(int)
    
    # Generate Breakout Signal
    df['breakout_signal'] = df['breakout_ratio'] * df['liquidity_flag']
    
    # Normalize by Cross-Sectional Rank
    df['factor'] = df['breakout_signal'].rolling(window=5, min_periods=1).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    return df['factor']
