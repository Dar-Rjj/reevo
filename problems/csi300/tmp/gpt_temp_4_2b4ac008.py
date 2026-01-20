import pandas as pd
import pandas as pd
from scipy.stats import skew

def heuristics_v2(data):
    # Measure Price Momentum
    short_term_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    long_term_momentum = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    combined_momentum = 0.7 * short_term_momentum + 0.3 * long_term_momentum
    
    # Calculate Price Skewness (5-day rolling window)
    skewness = data['close'].rolling(5).apply(lambda x: skew(x), raw=True)
    
    # Adjust Momentum by Skewness
    skewness_adjusted_momentum = combined_momentum * skewness
    skewness_adjusted_momentum = skewness_adjusted_momentum.clip(-2, 2)
    
    # Volume Confirmation
    volume_ma = data['volume'].rolling(5).mean()
    volume_ratio = data['volume'] / volume_ma
    
    # Calculate daily returns and volatility
    daily_returns = data['close'].pct_change()
    volatility = daily_returns.rolling(5).std()
    
    # Combine signals and normalize
    factor = skewness_adjusted_momentum * volume_ratio
    factor = factor / (volatility + 1e-6)  # Add small constant to avoid division by zero
    
    return factor
