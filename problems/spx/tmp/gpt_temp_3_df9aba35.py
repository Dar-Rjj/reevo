import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute daily price range
    daily_range = df['high'] - df['low']
    
    # Apply 5-day exponential moving average on daily range
    range_momentum = daily_range.ewm(span=5, adjust=False).mean()
    
    # Compute 3-day volume slope
    volume_slope = df['volume'].rolling(window=3).apply(lambda x: linregress(np.arange(3), x)[0])
    
    # Take the sign of the volume slope
    volume_trend = np.sign(volume_slope)
    
    # Combine signals multiplicatively
    combined_signal = range_momentum * volume_trend
    
    # Apply 3-day delay
    factor_values = combined_signal.shift(3)
    
    return factor_values
