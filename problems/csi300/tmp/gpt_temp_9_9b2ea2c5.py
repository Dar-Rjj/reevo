import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Directional Momentum (3-day return)
    df['return_3d'] = df['close'].pct_change(3)
    
    # Assess Momentum Consistency (count same-direction days in 5-day lookback)
    df['momentum_sign'] = np.sign(df['return_3d'])
    df['momentum_consistency'] = df['momentum_sign'].rolling(window=5, min_periods=1).apply(lambda x: len(x[x == x.iloc[-1]]), raw=False)
    
    # Calculate Recent Volatility (5-day std of Close)
    df['volatility_5d'] = df['close'].rolling(window=5, min_periods=1).std()
    
    # Normalize Momentum by Volatility
    df['momentum_vol_adj'] = df['return_3d'] / df['volatility_5d']
    
    # Compute Volume Confirmation (3-day volume Z-score)
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=3, min_periods=1).mean()) / df['volume'].rolling(window=3, min_periods=1).std()
    
    # Weight Momentum Score by Volume Confirmation
    df['factor'] = df['momentum_vol_adj'] * df['volume_zscore']
    
    return df['factor']
