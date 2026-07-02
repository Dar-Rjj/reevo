import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate daily returns
    returns = data['close'].pct_change()
    
    # Smoothed Momentum (EMA of returns with decay=0.5)
    smoothed_momentum = returns.ewm(alpha=0.5, adjust=False).mean()
    
    # Normalize smoothed momentum cross-sectionally
    normalized_momentum = smoothed_momentum.groupby(smoothed_momentum.index).transform(lambda x: (x - x.mean()) / x.std())
    
    # Volatility Adjustment (10-day rolling std of returns)
    rolling_volatility = returns.rolling(window=10, min_periods=1).std()
    
    # Combine components (multiply normalized momentum by inverse volatility)
    # Add small constant to avoid division by zero
    factor = normalized_momentum * (1 / (rolling_volatility + 1e-6))
    
    return factor
