def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Measure Closing Momentum
    closing_momentum = (df['close'] - df['open']) / intraday_range.replace(0, 1e-9)
    
    # Compute Relative Volume
    rolling_volume_mean = df['volume'].rolling(window=5, min_periods=1).mean()
    relative_volume = df['volume'] / rolling_volume_mean.replace(0, 1e-9)
    
    # Combine Momentum and Volume
    volume_adjusted_momentum = closing_momentum * relative_volume
    
    # Apply 5-day Z-Score
    factor = volume_adjusted_momentum.rolling(window=5, min_periods=1).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0)
    
    return factor
