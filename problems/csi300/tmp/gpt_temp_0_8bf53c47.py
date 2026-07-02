import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day rolling linear regression slope for close price
    price_slope = df['close'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True
    )
    
    # Calculate 5-day rolling linear regression slope for volume
    volume_slope = df['volume'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True
    )
    
    # Normalize price slope by 20-day rolling standard deviation of close price
    price_std = df['close'].rolling(window=20).std()
    normalized_price_slope = price_slope / price_std
    
    # Normalize volume slope by 20-day rolling standard deviation of volume
    volume_std = df['volume'].rolling(window=20).std()
    normalized_volume_slope = volume_slope / volume_std
    
    # Weighted combination of normalized slopes
    divergence_factor = normalized_price_slope * normalized_volume_slope
    
    # Apply sigmoid scaling to the divergence factor
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    signal = divergence_factor.apply(sigmoid)
    
    return signal
