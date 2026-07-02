import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy of the input dataframe to avoid modifying the original
    data = df.copy()
    
    # Sentiment Strength calculation
    # Normalized rolling mean of volume (window=15)
    sentiment_strength = data['volume'].rolling(window=15, min_periods=1).mean()
    sentiment_strength = (sentiment_strength - sentiment_strength.min()) / (sentiment_strength.max() - sentiment_strength.min())
    
    # Volatility adjustment using rolling standard deviation of close prices (window=20)
    volatility = data['close'].rolling(window=20, min_periods=1).std()
    volatility_adjustment = 1 / (volatility + 1e-6)  # Add small constant to avoid division by zero
    
    # Combine sentiment strength with volatility adjustment
    sentiment_factor = sentiment_strength * volatility_adjustment
    
    # Price Reversal Signal calculation
    # Delta between current close and close 5 days ago
    delta = data['close'] - data['close'].shift(5)
    
    # Calculate returns
    returns = data['close'].pct_change()
    
    # Rolling rank of returns (window=10)
    return_rank = returns.rolling(window=10, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Normalize the delta using the return rank
    reversal_signal = delta * return_rank
    
    # Combine both factors with equal weight
    factor = 0.5 * sentiment_factor + 0.5 * reversal_signal
    
    return factor
