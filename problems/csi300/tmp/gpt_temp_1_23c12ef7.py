import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    close = data['close']
    volume = data['volume']
    
    # Calculate rolling price slope (10-day)
    price_slope = close.rolling(window=10).apply(
        lambda x: linregress(np.arange(len(x)), x)[0], raw=False
    )
    
    # Calculate rolling volume slope (10-day)
    volume_slope = volume.rolling(window=10).apply(
        lambda x: linregress(np.arange(len(x)), x)[0], raw=False
    )
    
    # Calculate rolling correlation (10-day) between price and volume
    rolling_corr = close.rolling(window=10).corr(volume)
    
    # Compute divergence factor
    divergence = (price_slope - volume_slope) * rolling_corr.abs()
    
    return divergence
