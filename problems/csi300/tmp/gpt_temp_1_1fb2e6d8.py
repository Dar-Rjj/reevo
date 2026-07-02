import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Calculate Momentum Decay Strength factor based on smoothed momentum and decay factor.
    
    Parameters:
    data (pd.DataFrame): DataFrame with market data (date index, columns: open, high, low, close, volume, amount)
    
    Returns:
    pd.Series: Factor values indexed by date
    """
    close = data['close']
    
    # Calculate daily returns
    returns = close.pct_change()
    
    # Rolling rank of returns (20-day window)
    def rolling_rank(series, window):
        return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
    
    ranked_returns = rolling_rank(returns, 20)
    
    # Cross-sectional normalization (Z-score)
    def normalize(series):
        return (series - series.mean()) / series.std()
    
    smoothed_momentum = normalize(ranked_returns)
    
    # Exponential decay factor
    def exponential_decay(window, decay_rate):
        weights = np.array([decay_rate ** (window - i - 1) for i in range(window)])
        weights = weights / weights.sum()
        return weights
    
    decay_window = 10
    decay_rate = 0.9
    weights = exponential_decay(decay_window, decay_rate)
    
    # Apply decay to smoothed momentum
    decayed_momentum = smoothed_momentum.rolling(decay_window).apply(
        lambda x: np.sum(x * weights), raw=True
    )
    
    # Final factor is the decayed momentum
    factor = decayed_momentum
    
    return factor
