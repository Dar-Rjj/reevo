def heuristics_v2(df):
    # Price Strength Component
    # Calculate rolling mean of close prices over 20 periods
    rolling_mean_close = df['close'].rolling(window=20, min_periods=1).mean()
    # Calculate ratio of current close to rolling mean
    price_strength = df['close'] / rolling_mean_close
    
    # Volume Divergence Component
    # Calculate rolling rank of volume over 10 periods
    volume_rank = df['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    # Calculate 1-period difference of volume
    volume_delta = df['volume'].diff(periods=1)
    
    # Combine components
    # Normalize price strength cross-sectionally (z-score)
    normalized_price_strength = (price_strength - price_strength.mean()) / price_strength.std()
    # Combine with volume divergence
    factor = normalized_price_strength * volume_rank * volume_delta
    
    return factor
