def heuristics_v2(df):
    # Momentum Score Calculation
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Calculate cumulative returns over 10-day window (Rule 3: only current and past data)
    momentum = returns.rolling(window=10, min_periods=1).sum()
    
    # Normalize momentum using z-score (cross-sectional)
    momentum_zscore = momentum.groupby(momentum.index).transform(lambda x: (x - x.mean()) / x.std())
    momentum_normalized = momentum_zscore.rank(pct=True)  # cross-sectional ranking
    
    # Liquidity Adjustment Calculation
    # Calculate volume signal (5-day rolling mean)
    volume_signal = df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Normalize volume signal using z-score (cross-sectional)
    volume_zscore = volume_signal.groupby(volume_signal.index).transform(lambda x: (x - x.mean()) / x.std())
    volume_normalized = volume_zscore.rank(pct=True)  # cross-sectional ranking
    
    # Combine momentum and liquidity signals
    factor = momentum_normalized * volume_normalized
    
    # Apply EMA smoothing with window 3
    factor_smoothed = factor.ewm(span=3, adjust=False).mean()
    
    return factor_smoothed
