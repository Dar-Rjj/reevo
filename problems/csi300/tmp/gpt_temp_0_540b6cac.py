def heuristics_v2(df):
    # Liquidity Change Signal
    # Delta of volume over 5 days
    delta_volume = df['volume'].diff(5)
    
    # Rolling rank of delta_volume over 20 days
    rolling_rank_delta = delta_volume.rolling(20).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    
    # Cross-sectional rank of rolling_rank_delta
    cross_rank = rolling_rank_delta.groupby(rolling_rank_delta.index).rank(pct=True)
    
    # Z-score of cross-sectional rank
    zscore = (cross_rank - cross_rank.rolling(20).mean()) / cross_rank.rolling(20).std()
    
    # Price Momentum Confirmation
    # SMA of close over 10 days
    sma_close = df['close'].rolling(10).mean()
    
    # Daily returns
    daily_returns = df['close'].pct_change(1)
    
    # Rolling std of returns over 10 days
    rolling_std = daily_returns.rolling(10).std()
    
    # Volatility adjusted return
    volatility_adjusted_return = daily_returns / rolling_std
    
    # Combine signals
    factor = zscore * volatility_adjusted_return
    
    return factor
