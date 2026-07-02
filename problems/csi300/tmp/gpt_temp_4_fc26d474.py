def heuristics_v2(data):
    # Calculate Intraday Return
    intraday_return = data['close'] - data['open']
    
    # Normalize by Range
    price_range = data['high'] - data['low']
    normalized_price = intraday_return / price_range
    
    # Calculate Volume Deviation
    volume_ma = data['volume'].rolling(window=5).mean()
    volume_deviation = data['volume'] - volume_ma
    
    # Create Divergence Signal
    divergence_signal = normalized_price * volume_deviation
    
    # Apply 3-day Rolling Z-Score
    factor = divergence_signal.rolling(window=3).apply(lambda x: (x[-1] - x.mean()) / x.std())
    
    return factor
