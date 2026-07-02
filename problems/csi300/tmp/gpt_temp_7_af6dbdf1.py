import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    # EMA of close with window 10
    ema = df['close'].ewm(span=10, adjust=False).mean()
    
    # Delta between current close and close 5 days ago
    delta = df['close'] - df['close'].shift(5)
    
    # Volume Confirmation Signal
    # Rolling rank of volume over 20 days
    rolling_rank = df['volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Z-score of volume compared to rolling mean of past 10 days
    rolling_mean = df['volume'].shift(1).rolling(10).mean()  # Using shift(1) to exclude current value
    rolling_std = df['volume'].shift(1).rolling(10).std()
    zscore = (df['volume'] - rolling_mean) / rolling_std
    
    # Combine components
    momentum_divergence = ema * delta
    volume_confirmation = rolling_rank * zscore
    
    # Final factor
    factor = momentum_divergence * volume_confirmation
    
    return factor
