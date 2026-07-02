import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Sentiment Score branch
    # Calculate delta(volume) and delta(close)
    delta_volume = data['volume'].diff()
    delta_close = data['close'].diff()
    
    # Z-score of deltas
    zscore_volume = (delta_volume - delta_volume.rolling(window=10, min_periods=1).mean()) / delta_volume.rolling(window=10, min_periods=1).std()
    zscore_close = (delta_close - delta_close.rolling(window=10, min_periods=1).mean()) / delta_close.rolling(window=10, min_periods=1).std()
    
    # Combine z-scores
    combined_zscore = zscore_volume + zscore_close
    
    # Normalize and cross-sectional rank
    sentiment_score = combined_zscore.rolling(window=10, min_periods=1).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Liquidity Shift Ratio branch
    # Calculate rolling mean and std of volume
    rolling_mean_volume = data['volume'].rolling(window=10, min_periods=1).mean()
    rolling_std_volume = data['volume'].rolling(window=5, min_periods=1).std()
    
    # Calculate ratio
    liquidity_ratio = rolling_mean_volume / (rolling_std_volume + 1e-6)  # Add small constant to avoid division by zero
    
    # Combine with Sentiment Score
    factor = sentiment_score * liquidity_ratio
    
    return factor
