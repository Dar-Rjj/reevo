def heuristics_v2(df):
    # Intraday Momentum Calculation
    intraday_momentum = (df['high'] - df['close']) / df['close']
    intraday_momentum_rolling = intraday_momentum.rolling(window=5, min_periods=1).mean()
    
    # Volume-Weighted Normalization
    volume_weighted_momentum = intraday_momentum_rolling * df['volume']
    
    # Normalize by High-Low Range
    high_low_range = (df['high'] - df['low']) / df['close']
    max_min_range = high_low_range.rolling(window=20, min_periods=1).apply(lambda x: x.max() - x.min())
    
    # Final Factor Calculation
    factor = volume_weighted_momentum / max_min_range
    
    return factor
