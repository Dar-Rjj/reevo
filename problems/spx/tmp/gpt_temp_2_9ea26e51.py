def heuristics_v2(df):
    # Calculate Price Momentum: 20-day rolling mean of close price
    momentum = df['close'].rolling(window=20, min_periods=1).mean()
    
    # Rank by Quantile: 20-day rolling quantile rank (0 to 1)
    quantile_rank = momentum.rolling(window=20, min_periods=1).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    
    return quantile_rank
