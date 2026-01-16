def heuristics_v2(df):
    # Daily Range Ratio
    df['Daily Range Ratio'] = (df['high'] - df['low']) / df['close']
    
    # Exponential Weighted Avg of Daily Range Ratio past 5 days
    df['EWMA Range Ratio'] = df['Daily Range Ratio'].ewm(span=5, adjust=False).mean()
    
    # Volume-Weighted Momentum
    df['Volume Weighted Momentum'] = (
        df['Daily Range Ratio'].rolling(window=5).apply(lambda x: (x * df.loc[x.index, 'volume']).sum()) /
        df['volume'].rolling(window=5).sum()
    )
    
    # Rolling Standard Deviation of Close past 20 days
    df['Rolling Std Close'] = df['close'].rolling(window=20).std()
    
    # Volatility Scaling
    df['Volatility Scaling'] = 1 / df['Rolling Std Close']
    
    # Compute Factor
    factor = df['EWMA Range Ratio'] * df['Volume Weighted Momentum'] * df['Volatility Scaling']
    
    return factor
