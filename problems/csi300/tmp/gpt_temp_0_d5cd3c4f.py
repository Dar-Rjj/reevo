import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Initialize output Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate Price Trend Component
    price_slopes = []
    for i in range(len(df)):
        if i < 4:  # Need at least 5 points for regression
            price_slopes.append(np.nan)
            continue
        window = df['close'].iloc[i-4:i+1]  # Current and past 4 days
        slope = linregress(range(5), window).slope
        normalized_slope = slope / df['close'].iloc[i]  # Normalize by current price
        price_slopes.append(normalized_slope)
    
    # Calculate Volume Trend Component
    volume_slopes = []
    for i in range(len(df)):
        if i < 4:  # Need at least 5 points for regression
            volume_slopes.append(np.nan)
            continue
        window = df['volume'].iloc[i-4:i+1]  # Current and past 4 days
        slope = linregress(range(5), window).slope
        volume_slopes.append(slope)
    
    # Combine components to create factor
    for i in range(len(df)):
        if i < 4:
            factor.iloc[i] = np.nan
        else:
            # Price-Volume Divergence calculation
            factor.iloc[i] = (price_slopes[i] * price_slopes[i]) - volume_slopes[i]
    
    return factor
