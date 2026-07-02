def heuristics_v2(df):
    # Calculate Short-Term Price Momentum
    df['momentum'] = df['close'] / df['close'].shift(5) - 1
    
    # Calculate Volume Surprise
    df['rolling_mean_volume'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_surprise'] = df['volume'] / df['rolling_mean_volume']
    
    # Adjust Momentum by Volume Surprise
    df['momentum_adjusted'] = df['momentum'] * df['volume_surprise']
    
    # Normalize Across Universe using Cross-Sectional Z-Score
    df['z_score'] = df.groupby(df.index)['momentum_adjusted'].transform(lambda x: (x - x.mean()) / x.std())
    
    # Weight by Historical Volatility
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10, min_periods=1).std()
    
    # Combine Signals
    df['factor'] = df['z_score'] * df['volatility']
    
    return df['factor']
