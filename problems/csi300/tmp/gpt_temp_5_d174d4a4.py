import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Price Efficiency Component
    # Calculate ratio of current close to rolling median (10 periods)
    rolling_median = df['close'].rolling(window=10, min_periods=1).median()
    ratio = df['close'] / rolling_median
    
    # Normalize using cross-sectional rank
    price_efficiency = ratio.rank(pct=True)
    
    # Liquidity Skewness Component
    # Calculate volume delta
    volume_delta = df['volume'] - df['volume'].shift(1)
    
    # Calculate rolling skewness (15 periods)
    def rolling_skew(x):
        return x.skew()
    
    volume_skewness = volume_delta.rolling(window=15, min_periods=1).apply(rolling_skew)
    
    # Calculate z-score of log transformed amount
    log_amount = np.log1p(df['amount'])
    zscore = (log_amount - log_amount.rolling(window=1, min_periods=1).mean()) / log_amount.rolling(window=1, min_periods=1).std()
    
    # Combine components with equal weight
    liquidity_skewness = 0.5 * volume_skewness + 0.5 * zscore
    
    # Final factor is average of both components
    factor = 0.5 * price_efficiency + 0.5 * liquidity_skewness
    
    return factor
