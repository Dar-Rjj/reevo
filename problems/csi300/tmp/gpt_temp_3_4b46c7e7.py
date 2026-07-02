import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Calculate Price-Volume Divergence Factor based on 5-day trends of price and volume.
    Positive factor indicates bullish divergence (price down + volume up),
    Negative factor indicates bearish divergence (price up + volume down).
    """
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling 5-day price and volume trends
    for i in range(4, len(data)):
        window = data.iloc[i-4:i+1]  # current and past 4 days (5-day window)
        
        # Calculate price trend (slope of close prices)
        price_slope = linregress(range(5), window['close']).slope
        
        # Calculate volume trend (slope of volume)
        volume_slope = linregress(range(5), window['volume']).slope
        
        # Determine divergence
        if price_slope > 0 and volume_slope < 0:
            # Bearish divergence (price up, volume down)
            factor.iloc[i] = -1.0
        elif price_slope < 0 and volume_slope > 0:
            # Bullish divergence (price down, volume up)
            factor.iloc[i] = 1.0
        else:
            # No clear divergence
            factor.iloc[i] = 0.0
    
    return factor
