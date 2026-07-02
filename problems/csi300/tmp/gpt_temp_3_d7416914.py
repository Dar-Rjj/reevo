import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day Price Return
    price_return = df['close'] / df['close'].shift(5) - 1
    
    # Compute 3-day Volume Slope
    volume_slope = df['volume'].rolling(window=3).apply(
        lambda x: linregress(np.arange(3), x)[0],
        raw=True
    )
    
    # Normalize Volume Slope by 10-day Volume Mean
    volume_mean = df['volume'].rolling(window=10).mean()
    normalized_slope = volume_slope / volume_mean
    
    # Combine Momentum and Acceleration
    factor = price_return * normalized_slope
    
    return factor.dropna()
