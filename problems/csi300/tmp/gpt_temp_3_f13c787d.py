import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate Price Variation
    price_variation = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume Variation
    volume_variation = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Combine Variations
    combined_variation = price_variation * volume_variation
    
    # Apply z-score normalization
    normalized_variation = combined_variation.groupby(df.index.date).apply(zscore)
    normalized_variation = normalized_variation.reset_index(level=0, drop=True)
    
    # Weight by Magnitude
    weighted_variation = normalized_variation * df['open']
    
    # Scale by 5-day rolling std of Volume
    rolling_std_volume = df['volume'].rolling(window=5).std()
    factor = weighted_variation / rolling_std_volume
    
    return factor.dropna()
