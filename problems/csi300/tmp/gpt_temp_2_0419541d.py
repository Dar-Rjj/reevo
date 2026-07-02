import numpy as np
def heuristics_v2(df):
    # Compute Intraday Return
    intraday_return = df['close'] - df['open']
    
    # Normalize by Price Range
    price_range = df['high'] - df['low']
    normalized_return = intraday_return / price_range
    
    # Calculate Volume Percentile
    volume_percentile = df['volume'].rolling(window=20).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    
    # Apply Sigmoid Transformation
    sigmoid_weight = 1 / (1 + np.exp(-volume_percentile))
    
    # Combine Signals
    factor = normalized_return * sigmoid_weight
    
    # Apply 5-day Rolling Mean
    smoothed_factor = factor.rolling(window=5).mean()
    
    return smoothed_factor
