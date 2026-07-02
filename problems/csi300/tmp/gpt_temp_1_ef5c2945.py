def heuristics_v2(data):
    # Price Impact Ratio
    # Calculate absolute price change
    delta_close = (data['close'] - data['close'].shift(1)).abs()
    
    # Calculate rolling mean of volume (window=5)
    rolling_volume = data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Compute ratio
    price_impact_ratio = delta_close / rolling_volume
    
    # Normalize cross-sectionally
    price_impact_ratio_norm = price_impact_ratio.groupby(price_impact_ratio.index).transform(lambda x: (x - x.mean()) / x.std())
    
    # Order Flow Divergence
    # Calculate EMA of high and low (span=3)
    ema_high = data['high'].ewm(span=3, adjust=False).mean()
    ema_low = data['low'].ewm(span=3, adjust=False).mean()
    
    # Compute difference between EMAs
    order_flow_divergence = ema_high - ema_low
    
    # Calculate rolling rank (window=10)
    order_flow_rank = order_flow_divergence.rolling(window=10, min_periods=1).apply(lambda x: x.rank(pct=True)[-1])
    
    # Combine factors with equal weight
    factor = 0.5 * price_impact_ratio_norm + 0.5 * order_flow_rank
    
    return factor
