import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Price Trend Strength
    price_slope = df['close'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x)[0], raw=True)
    
    # Calculate Volume Trend Strength
    volume_slope = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x)[0], raw=True)
    
    # Compute Divergence Signal
    divergence_signal = -1 * (price_slope * volume_slope)
    
    return divergence_signal
