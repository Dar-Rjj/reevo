def heuristics_v2(df):
    # Calculate Price Momentum
    df['momentum'] = df['close'] - df['close'].shift(5)
    
    # Calculate Daily Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate 5-day Volume Moving Average
    df['volume_ma'] = df['volume'].rolling(window=5).mean()
    
    # Adjust Price Range by Volume MA
    df['adjusted_range'] = df['price_range'] * df['volume_ma']
    
    # Normalize Momentum by Adjusted Range and multiply by Close price
    df['normalized_momentum'] = (df['momentum'] / df['adjusted_range']) * df['close']
    
    # Apply Rolling Z-Score with a 20-day window
    df['factor'] = df['normalized_momentum'].rolling(window=20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std(), raw=False)
    
    return df['factor']
