import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = (df['high'] / df['low'].shift(1) - 1).fillna(0)
    
    # Smoothed Momentum with EMA (decay=0.6)
    ema_momentum = returns.ewm(alpha=1-0.6, adjust=False).mean()
    
    # Normalize using cross-sectional z-score
    normalized_momentum = (ema_momentum - ema_momentum.mean()) / ema_momentum.std()
    
    # Decay Factor with exponential decay (window=15, decay_rate=0.8)
    decay_factor = pd.Series(np.zeros(len(df)), index=df.index)
    for i in range(1, len(df)):
        if i < 15:
            window = i
        else:
            window = 15
        weights = np.array([0.8 ** (window - j - 1) for j in range(window)])
        weights = weights / weights.sum()
        decay_factor.iloc[i] = (normalized_momentum.iloc[i-window:i] * weights).sum()
    
    # Momentum Decay Strength
    factor = normalized_momentum * decay_factor
    
    return factor
