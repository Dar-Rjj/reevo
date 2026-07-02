import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day price slope
    price_slopes = df['close'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x).slope, raw=True)
    
    # Calculate 5-day volume slope
    volume_slopes = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x).slope, raw=True)
    
    # Compute divergence factor
    divergence_factor = -price_slopes * volume_slopes
    
    return divergence_factor
