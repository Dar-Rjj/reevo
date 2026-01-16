def heuristics_v2(df):
    # Raw Intraday Momentum
    df['momentum'] = (df['high'] - df['low']) / df['close']
    
    # Volume Adjustment
    df['rolling_volume_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['rolling_volume_avg']
    
    # Adjusted Momentum
    df['adjusted_momentum'] = df['momentum'] * df['volume_ratio']
    
    # Price Stability Filter
    df['rolling_high'] = df['high'].rolling(window=3).max()
    df['rolling_low'] = df['low'].rolling(window=3).min()
    df['price_range'] = (df['rolling_high'] - df['rolling_low']) / df['close']
    
    # Apply Stability Filter
    df['stability_factor'] = df['price_range'].apply(lambda x: 0.8 if x > df['price_range'].quantile(0.75) else 1.2)
    
    # Combined Signal
    df['signal'] = df['adjusted_momentum'] * df['stability_factor']
    
    return df['signal']
