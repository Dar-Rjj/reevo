def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = df['close'] / df['open']
    
    # Calculate 5-day Rolling Std of Returns
    rolling_std = df['close'].pct_change().rolling(window=5).std()
    
    # Normalize Intraday Return by Historical Volatility
    normalized_efficiency = intraday_return / rolling_std
    
    # Calculate Volume Percentile
    volume_percentile = df['volume'].rolling(window=20).apply(lambda x: (x.rank().iloc[-1] / len(x)), raw=False)
    
    # Combine Components
    factor = normalized_efficiency * volume_percentile
    
    # Apply 3-day Exponential Moving Average
    factor_smoothed = factor.ewm(span=3, adjust=False).mean()
    
    return factor_smoothed
