import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Dynamic volatility scaling with asymmetric treatment for up/down moves
    up_volatility = (df['high'] - df['open']).rolling(5).std()
    down_volatility = (df['open'] - df['low']).rolling(5).std()
    directional_volatility = np.where(df['close'] > df['open'], up_volatility, down_volatility) + 1e-7
    
    # Volume-confirmed momentum with nonlinear scaling
    raw_momentum = (df['close'] - df['open'].rolling(3).mean())
    volume_confirmation = np.sign(raw_momentum) * np.log1p(df['volume'].rolling(5).mean())
    momentum = raw_momentum / directional_volatility * volume_confirmation
    
    # Adaptive mean reversion using volatility regimes
    volatility_regime = (df['high'] - df['low']).rolling(10).std()
    mean_reversion = (df['close'] - df['close'].rolling(5).mean()) 
    mean_reversion = np.where(volatility_regime > volatility_regime.median(), 
                             mean_reversion / (directional_volatility * 2),
                             mean_reversion / directional_volatility)
    
    # Time decay weighted combination
    decay_factor = np.exp(-np.arange(5)/2.0)[::-1]  # Half-life of 2 periods
    recent_momentum = (df['close'] - df['open']).rolling(5).apply(lambda x: np.sum(x * decay_factor))
    alpha = momentum * 0.6 + mean_reversion * 0.3 + recent_momentum * 0.1
    
    return alpha
