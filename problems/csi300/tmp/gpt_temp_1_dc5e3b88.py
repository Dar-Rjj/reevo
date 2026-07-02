import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Calculate daily returns
    returns = np.log(data['close'] / data['close'].shift(1))
    
    # Calculate rolling rank of returns with window=20 (using only past data)
    rank_returns = returns.rolling(window=20, min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Normalize the ranked returns cross-sectionally
    normalized_returns = (rank_returns - rank_returns.expanding().mean()) / rank_returns.expanding().std()
    
    # Calculate exponential decay factor with window=10 and decay_rate=0.9
    decay_window = 10
    decay_rate = 0.9
    weights = np.array([decay_rate ** i for i in range(decay_window)][::-1])
    
    def exponential_decay(series):
        if len(series) >= decay_window:
            return np.sum(series[-decay_window:] * weights) / np.sum(weights)
        else:
            return np.nan
    
    decay_factor = normalized_returns.rolling(window=decay_window, min_periods=1).apply(exponential_decay)
    
    # Calculate Momentum Decay Strength by multiplying smoothed momentum with decay factor
    factor = normalized_returns * decay_factor
    
    return factor
