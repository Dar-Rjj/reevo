import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate Price Momentum Score
    # Ratio of current close to 15-day rolling mean (using past 10 to t-1 for current day)
    rolling_mean_close = data['close'].shift(10).rolling(window=15, min_periods=1).mean()
    momentum_ratio = data['close'] / rolling_mean_close
    
    # Cross-sectional rank of momentum ratio
    momentum_rank = momentum_ratio.groupby(momentum_ratio.index).rank(pct=True)
    
    # Calculate Liquidity Imbalance
    # Ratio of current volume to 20-day rolling mean (using past 5 to t-1 for current day)
    rolling_mean_volume = data['volume'].shift(5).rolling(window=20, min_periods=1).mean()
    liquidity_ratio = data['volume'] / rolling_mean_volume
    
    # Cross-sectional normalization of liquidity ratio
    liquidity_norm = liquidity_ratio.groupby(liquidity_ratio.index).apply(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Combine factors with equal weight
    factor = 0.5 * momentum_rank + 0.5 * liquidity_norm
    
    return factor
