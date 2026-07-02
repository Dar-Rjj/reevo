def heuristics_v2(df):
    # Momentum Divergence Score
    # Relative Strength Comparison
    ema10 = df['close'].ewm(span=10, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    
    # Rank the EMAs cross-sectionally (within each day)
    rank_ema10 = ema10.groupby(ema10.index).rank(pct=True)
    rank_ema20 = ema20.groupby(ema20.index).rank(pct=True)
    
    # Calculate divergence as difference between ranks
    divergence = rank_ema10 - rank_ema20
    
    # Normalize using cross-sectional z-score
    momentum_score = divergence.groupby(divergence.index).apply(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Liquidity Adjustment
    # Volume Stability (coefficient of variation)
    vol_std = df['volume'].rolling(5).std()
    vol_mean = df['volume'].rolling(5).mean()
    vol_stability = vol_std / (vol_mean + 1e-6)  # Add small constant to avoid division by zero
    
    # Invert stability (higher stability -> higher weight)
    liquidity_weight = 1 / (vol_stability + 1e-6)
    
    # Combine momentum score with liquidity adjustment
    factor = momentum_score * liquidity_weight
    
    return factor
