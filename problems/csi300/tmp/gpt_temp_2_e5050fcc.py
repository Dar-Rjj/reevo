def heuristics_v2(data):
    # Calculate Volume Z-Score
    volume_mean = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_std = data['volume'].rolling(window=20, min_periods=10).std()
    volume_z = (data['volume'] - volume_mean) / volume_std
    
    # Calculate Recent Price Change normalized by price level
    price_change = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Cross-sectional ranking
    volume_efficiency_rank = volume_z.groupby(data.index).rank(pct=True)
    price_change_rank = price_change.groupby(data.index).rank(pct=True)
    
    # Combine ranks and standardize
    combined_rank = (volume_efficiency_rank - price_change_rank)
    factor = combined_rank.groupby(data.index).transform(lambda x: (x - x.mean()) / x.std())
    
    return factor
