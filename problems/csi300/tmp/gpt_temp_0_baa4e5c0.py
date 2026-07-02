import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Calculate Price Slope using linear regression over 5 days
    def price_slope(series):
        return linregress(np.arange(5), series[-5:])[0]
    
    # Calculate Volume Slope using linear regression over 5 days
    def volume_slope(series):
        return linregress(np.arange(5), series[-5:])[0]
    
    # Compute Price Slope for each day
    price_slopes = data['close'].rolling(window=5).apply(price_slope, raw=True)
    
    # Compute Volume Slope for each day
    volume_slopes = data['volume'].rolling(window=5).apply(volume_slope, raw=True)
    
    # Compute Divergence: Subtract Volume Slope from Price Slope and multiply by Price Slope Sign
    divergence = (price_slopes - volume_slopes) * np.sign(price_slopes)
    
    return divergence
