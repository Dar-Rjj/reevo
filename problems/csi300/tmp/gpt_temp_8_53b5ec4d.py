def heuristics_v2(df):
    # Momentum Divergence Component
    # EMA 10-day
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    # Rank over 20 days (current and past 19)
    rank_ema_20 = ema_10.rolling(window=20).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    
    # EMA 5-day
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    # Rank over 10 days (current and past 9)
    rank_ema_10 = ema_5.rolling(window=10).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    
    momentum_divergence = rank_ema_20 - rank_ema_10
    
    # Liquidity Adjustment Component
    # Normalized Volume (current / 20-day mean)
    volume_mean_20 = df['volume'].rolling(window=20).mean()
    normalized_volume = df['volume'] / volume_mean_20
    
    # Volatility (10-day std of returns)
    returns = df['close'].pct_change()
    volatility_10 = returns.rolling(window=10).std()
    
    # Combine components with liquidity adjustment
    factor = momentum_divergence * normalized_volume * volatility_10
    return factor
