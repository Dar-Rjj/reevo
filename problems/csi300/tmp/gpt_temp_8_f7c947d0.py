import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Momentum Consistency - Correlation Component
    df['return_t'] = df['close'].pct_change()
    df['return_t-5'] = df['close'].pct_change(5)
    
    # EMA calculation for return_t and return_t-5 with decay=0.3
    alpha = 0.3
    df['ema_return_t'] = df['return_t'].ewm(alpha=alpha, adjust=False).mean()
    df['ema_return_t-5'] = df['return_t-5'].ewm(alpha=alpha, adjust=False).mean()
    
    # Correlation between EMA of return_t and EMA of return_t-5
    momentum_consistency = df['ema_return_t'].rolling(window=10).corr(df['ema_return_t-5'])
    
    # Normalize using cross-sectional rank
    momentum_factor = momentum_consistency.rolling(window=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Volatility Dampening Component
    rolling_std_close = df['close'].rolling(window=10).std()
    rolling_std_high = df['high'].rolling(window=10).std()
    volatility_ratio = rolling_std_close / rolling_std_high
    
    # Log transform of volume
    log_volume = df['volume'].apply(lambda x: np.log(x + 1))
    
    # Combine factors
    factor = momentum_factor - volatility_ratio + log_volume
    
    return factor
