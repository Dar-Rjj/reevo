def heuristics_v2(df):
    # Short-term Momentum: (Close{t} - Close{t-5}) / Close{t-5}
    short_term_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Intraday Momentum: (Close{t} - Open{t}) / Open{t}
    intraday_momentum = (df['close'] - df['open']) / df['open']
    
    # Volume Ratio: Volume{t} / Mean(Volume{t-4:t})
    volume_ratio = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Combine the two momentum components
    momentum = short_term_momentum + intraday_momentum
    
    # Volume-Adjusted Momentum with Volatility Normalization
    factor = momentum * volume_ratio
    
    return factor
