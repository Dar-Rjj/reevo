import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Deviation Anomaly - Deviation Amplitude
    # Calculate rolling mean of close prices (10-day window, lagged by 5 days)
    rolling_mean_close = df['close'].shift(5).rolling(window=10, min_periods=1).mean()
    
    # Calculate absolute deviation ratio
    deviation_ratio = (df['close'] - rolling_mean_close).abs() / rolling_mean_close
    
    # Normalize and rank cross-sectionally
    deviation_factor = deviation_ratio.groupby(df.index).rank(pct=True)
    
    # Liquidity Confirmation
    # Calculate rolling mean of volume (10-day window, lagged by 5 days)
    rolling_mean_volume = df['volume'].shift(5).rolling(window=10, min_periods=1).mean()
    
    # Calculate volume ratio
    volume_ratio = df['volume'] / rolling_mean_volume
    
    # Calculate spread adjustment (high-low spread)
    spread = df['high'] - df['low']
    
    # Normalize spread using cross-sectional z-score
    spread_zscore = spread.groupby(df.index).apply(
        lambda x: (x - x.mean()) / x.std()
    ).reset_index(level=0, drop=True)
    
    # Combine factors with weights (adjust weights as needed)
    combined_factor = (
        0.6 * deviation_factor + 
        0.3 * volume_ratio + 
        0.1 * spread_zscore
    )
    
    return combined_factor
