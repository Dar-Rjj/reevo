import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Divergence
    ema10 = df['close'].ewm(span=10, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    momentum_diff = ema10 - ema20
    
    # Normalize by rolling std
    rolling_std = df['close'].rolling(window=10).std()
    normalized_momentum = momentum_diff / rolling_std
    
    # Liquidity Adjustment components
    # Volume Z-score
    volume_mean = df['volume'].rolling(window=20).mean()
    volume_std = df['volume'].rolling(window=20).std()
    volume_zscore = (df['volume'] - volume_mean) / volume_std
    
    # Price Range Ratio
    price_range = (df['high'] - df['low']) / df['close']
    price_range_rank = price_range.rolling(window=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Multiply components
    liquidity_multiplier = volume_zscore * price_range_rank
    
    # Decay Factor (linear decay over 5 days)
    decay = np.linspace(1, 0.2, 5)
    decay_factor = liquidity_multiplier.rolling(window=5).apply(lambda x: (x * decay).sum() / decay.sum())
    
    # Combine components
    factor = normalized_momentum * decay_factor
    
    return factor
