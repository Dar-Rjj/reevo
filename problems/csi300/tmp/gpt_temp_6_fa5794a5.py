import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day Price Slope
    price_slope = df['close'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x)[0], raw=True)
    
    # Calculate 5-day Volume Slope
    volume_slope = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x)[0], raw=True)
    
    # Compute Divergence
    divergence = price_slope / volume_slope
    
    # Apply rolling Z-score normalization with a 20-day window
    rolling_mean = divergence.rolling(window=20).mean()
    rolling_std = divergence.rolling(window=20).std()
    divergence_zscore = (divergence - rolling_mean) / rolling_std
    
    return divergence_zscore
