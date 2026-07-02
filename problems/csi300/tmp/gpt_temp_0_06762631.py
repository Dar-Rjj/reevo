import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 3-day linear regression slope for price and volume
    def calculate_slope(series, window=3):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(len(y))
            slope = linregress(x, y).slope
            slopes.iloc[i] = slope
        return slopes
    
    # Calculate price trend (3-day slope)
    price_slope = calculate_slope(df['close'])
    
    # Calculate volume trend (3-day slope)
    volume_slope = calculate_slope(df['volume'])
    
    # Normalize price trend
    price_norm = price_slope / df['close'].shift(3) * 100
    price_norm = price_norm.fillna(0)
    
    # Normalize volume trend
    volume_norm = volume_slope / df['volume'].shift(3) * 100
    volume_norm = volume_norm.fillna(0)
    
    # Generate signals
    signal = pd.Series(0, index=df.index)
    
    # Positive divergence (price up, volume down)
    pos_cond = (price_norm > 1) & (volume_norm < -0.5)
    signal[pos_cond] = 1
    
    # Negative divergence (price down, volume up)
    neg_cond = (price_norm < -1) & (volume_norm > 0.5)
    signal[neg_cond] = -1
    
    return signal
