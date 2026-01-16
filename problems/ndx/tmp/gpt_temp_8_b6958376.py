def heuristics_v2(df):
    # Calculate Trend Momentum Component
    # Price Momentum: (Close(t) - Mean(Close(t-4) to Close(t))) / Close(t)
    close_ma4 = df['close'].rolling(window=5, min_periods=5).mean()
    price_momentum = (df['close'] - close_ma4) / df['close']
    
    # Intraday Efficiency: (Close(t) - Open(t)) / (High(t) - Low(t))
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Combine Trend Momentum components
    trend_momentum = price_momentum * intraday_efficiency
    
    # Calculate Volume Adjustment components
    # Volume Momentum: Volume(t) / Volume(t-1) - 1
    volume_momentum = df['volume'] / df['volume'].shift(1) - 1
    
    # Volume Stability: Volume(t) / Mean(Volume(t-4) to Volume(t))
    volume_ma4 = df['volume'].rolling(window=5, min_periods=5).mean()
    volume_stability = df['volume'] / volume_ma4
    
    # Combine Volume Adjustment components
    volume_adjustment = volume_momentum * volume_stability
    
    # Combine all components
    combined_factor = trend_momentum * volume_adjustment
    
    # Cross-sectional normalization (within each day)
    normalized_factor = combined_factor.groupby(combined_factor.index).transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    return normalized_factor
