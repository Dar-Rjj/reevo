import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum Divergence
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    momentum_divergence = ema_10 - ema_20
    
    # Normalize Momentum Divergence
    rolling_std = df['close'].rolling(window=10).std()
    normalized_momentum_divergence = momentum_divergence / rolling_std
    
    # Volume Z-Score
    rolling_mean_volume = df['volume'].rolling(window=20).mean()
    volume_z_score = (df['volume'] - rolling_mean_volume) / df['volume'].rolling(window=20).std()
    
    # Turnover Ratio
    rolling_mean_amount = df['amount'].rolling(window=20).mean()
    turnover_ratio = df['amount'] / rolling_mean_amount
    turnover_rank = turnover_ratio.rolling(window=10).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    
    # Liquidity Adjustment
    liquidity_adjustment = volume_z_score * turnover_rank
    
    # Decay Factor
    decay_factor = np.exp(-np.arange(5) / 5.0)
    decay_factor = decay_factor[::-1]
    liquidity_adjustment_decay = liquidity_adjustment.rolling(window=5).apply(lambda x: (x * decay_factor).sum(), raw=False)
    
    # Liquidity-Adjusted Momentum Divergence
    factor = normalized_momentum_divergence * liquidity_adjustment_decay
    
    return factor
