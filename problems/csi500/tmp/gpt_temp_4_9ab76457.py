import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Volatility Expansion: Compare current TR to rolling median TR (20-day window)
    tr_median_20d = tr.rolling(window=20, min_periods=10).median()
    volatility_expansion = tr / tr_median_20d
    
    # Liquidity Analysis
    dollar_volume = df['close'] * df['volume']
    liquidity_ratio = df['volume'] / df['amount']
    
    # Normalize liquidity metrics
    dollar_volume_z = (dollar_volume - dollar_volume.rolling(window=20, min_periods=10).mean()) / dollar_volume.rolling(window=20, min_periods=10).std()
    liquidity_ratio_z = (liquidity_ratio - liquidity_ratio.rolling(window=20, min_periods=10).mean()) / liquidity_ratio.rolling(window=20, min_periods=10).std()
    
    # Composite liquidity score (higher is better)
    liquidity_score = 0.6 * dollar_volume_z + 0.4 * liquidity_ratio_z
    
    # Momentum Continuation
    returns = df['close'].pct_change()
    momentum_acceleration = returns.diff()  # Rate of change of returns
    momentum_persistence = returns.rolling(window=5, min_periods=3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0)
    
    # Normalize momentum components
    momentum_acceleration_z = (momentum_acceleration - momentum_acceleration.rolling(window=20, min_periods=10).mean()) / momentum_acceleration.rolling(window=20, min_periods=10).std()
    momentum_persistence_z = (momentum_persistence - momentum_persistence.rolling(window=20, min_periods=10).mean()) / momentum_persistence.rolling(window=20, min_periods=10).std()
    
    # Composite momentum continuation strength
    momentum_strength = 0.5 * momentum_acceleration_z + 0.5 * momentum_persistence_z
    
    # Synthesize Composite Factor
    volatility_liquidity = volatility_expansion * liquidity_score
    factor = volatility_liquidity * momentum_strength
    
    return factor
