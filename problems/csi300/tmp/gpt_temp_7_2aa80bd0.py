import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Liquidity Ratio component
    volume = data['volume']
    rolling_mean_volume = volume.rolling(window=20, min_periods=10).mean()
    liquidity_ratio = volume / rolling_mean_volume
    
    # Cross-sectional rank and z-score of liquidity ratio
    ranked_liquidity = liquidity_ratio.groupby(level=0).rank(pct=True)
    zscore_liquidity = ranked_liquidity.groupby(level=0).apply(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Decay-Adjusted Momentum component
    close = data['close']
    returns = close.pct_change()
    
    # Exponential decay weights with half-life of 5
    decay_weights = np.array([0.5 ** (i/5) for i in range(10)])
    decay_weights = decay_weights / decay_weights.sum()  # normalize
    
    # Apply decay weights to past returns
    decay_adjusted_momentum = returns.rolling(window=10, min_periods=5).apply(
        lambda x: np.sum(x * decay_weights[:len(x)][::-1])
    )
    
    # Volatility scaling
    rolling_std = returns.rolling(window=10, min_periods=5).std()
    volatility_scaled_momentum = decay_adjusted_momentum / (rolling_std + 0.2)
    
    # Combine components
    factor = zscore_liquidity + volatility_scaled_momentum
    
    return factor
