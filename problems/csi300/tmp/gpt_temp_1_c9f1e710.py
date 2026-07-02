def heuristics_v2(df):
    # Calculate EMA of close price with a span of 10
    ema_t = df['close'].ewm(span=10, adjust=False).mean()
    
    # Calculate EMA of close price 5 days ago with a span of 10
    ema_t_minus_5 = df['close'].shift(5).ewm(span=10, adjust=False).mean()
    
    # Calculate Momentum Divergence as the difference between EMAs
    momentum_divergence = ema_t - ema_t_minus_5
    
    # Calculate rolling rank of Momentum Divergence over a window of 20
    rolling_rank = momentum_divergence.rolling(window=20).apply(lambda x: x.rank(pct=True).iloc[-1])
    
    # Calculate z-score of the rolling rank
    zscore = (rolling_rank - rolling_rank.rolling(window=20).mean()) / rolling_rank.rolling(window=20).std()
    
    # Calculate rolling mean of volume over a window of 20, shifted by 10 days
    rolling_mean_volume = df['volume'].shift(10).rolling(window=20).mean()
    
    # Calculate Liquidity Confirmation as the ratio of current volume to rolling mean volume
    liquidity_confirmation = df['volume'] / rolling_mean_volume
    
    # Normalize the absolute returns
    normalized_abs_return = df['close'].pct_change().abs().rolling(window=20).apply(lambda x: x / x.sum())
    
    # Multiply Momentum Divergence by normalized absolute returns
    momentum_with_liquidity_confirmation = momentum_divergence * normalized_abs_return
    
    # Final factor is the z-score multiplied by liquidity confirmation
    factor = zscore * liquidity_confirmation
    
    return factor
