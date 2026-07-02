import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day Price Slope using Close prices
    price_slope = df['close'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True
    )
    
    # Calculate 5-day Volume Slope using Volume
    volume_slope = df['volume'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True
    )
    
    # Compute Divergence by multiplying Price Slope by Volume Slope and taking negative sign
    divergence_factor = -price_slope * volume_slope
    
    return divergence_factor
