import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Short-Term Reversal Component
    # Calculate 5-day rolling mean of close prices
    rolling_mean = data['close'].rolling(window=5, min_periods=1).mean()
    # Calculate ratio of current close to rolling mean
    reversal_ratio = data['close'] / rolling_mean
    # Cross-sectional rank normalization
    reversal_factor = reversal_ratio.groupby(data.index).rank(pct=True)
    
    # Volume Confirmation
    # Calculate EMA of volume with window 10
    volume_ema = data['volume'].ewm(span=10, adjust=False).mean()
    # Calculate volume momentum factor
    volume_momentum = data['volume'] / volume_ema
    # Cross-sectional z-score normalization
    volume_factor = volume_momentum.groupby(data.index).apply(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Calculate rolling correlation between volume and volume momentum factor
    rolling_corr = data['volume'].rolling(window=10).corr(volume_momentum)
    
    # Combine components with EMA decay
    decay_factor = rolling_corr * reversal_factor
    # Apply EMA decay with alpha=0.2 and window=5
    final_factor = decay_factor.ewm(alpha=0.2, adjust=False).mean()
    
    return final_factor
