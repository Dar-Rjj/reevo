import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Liquidity Ratio component
    # Calculate 20-day rolling mean of volume
    rolling_volume = df['volume'].rolling(window=20, min_periods=10).mean()
    
    # Calculate ratio of current volume to rolling mean
    volume_ratio = df['volume'] / rolling_volume
    
    # Z-score the volume ratio cross-sectionally
    liquidity_zscore = volume_ratio.groupby(volume_ratio.index).transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Microstructure-Adjusted Momentum component
    # Calculate EMA of returns with span=10
    returns = df['close'].pct_change()
    ema_momentum = returns.ewm(span=10, min_periods=5).mean()
    
    # Calculate rolling mean of high-low spread (10-day window)
    spread = df['high'] - df['low']
    rolling_spread = spread.rolling(window=10, min_periods=5).mean()
    
    # Scale the momentum by spread volatility
    spread_scaling = 0.1 / (rolling_spread + 1e-6)  # small constant to avoid division by zero
    adjusted_momentum = ema_momentum * spread_scaling
    
    # Combine both components
    factor = liquidity_zscore + adjusted_momentum
    
    return factor
