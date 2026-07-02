import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Calculate price trends
    close = data['close']
    short_term_price_slope = close.rolling(window=5).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=True)
    long_term_price_slope = close.rolling(window=20).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=True)
    
    # Calculate volume trends
    volume = data['volume']
    short_term_volume_slope = volume.rolling(window=5).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=True)
    long_term_volume_slope = volume.rolling(window=20).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=True)
    
    # Calculate divergence components
    short_term_divergence = short_term_price_slope * short_term_volume_slope
    long_term_divergence = long_term_price_slope * long_term_volume_slope
    
    # Calculate divergence signal
    divergence_signal = short_term_divergence - long_term_divergence
    
    # Normalize signal
    price_std = close.rolling(window=20).std()
    normalized_signal = divergence_signal / price_std
    
    # Apply sigmoid transformation
    factor = 1 / (1 + np.exp(-normalized_signal))
    
    return factor
