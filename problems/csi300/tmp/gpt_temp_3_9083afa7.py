import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate 5-day price slope (using close prices)
    price_slopes = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data['close'].iloc[i-4:i+1]
        slope = linregress(np.arange(5), window.values).slope
        price_slopes.iloc[i] = slope
    
    # Calculate 5-day volume slope
    volume_slopes = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data['volume'].iloc[i-4:i+1]
        slope = linregress(np.arange(5), window.values).slope
        volume_slopes.iloc[i] = slope
    
    # Calculate rolling correlation between price and volume slopes (5-day window)
    correlations = pd.Series(index=data.index, dtype=float)
    for i in range(9, len(data)):  # Need at least 5 slopes to calculate correlation
        price_window = price_slopes.iloc[i-4:i+1]
        volume_window = volume_slopes.iloc[i-4:i+1]
        corr = np.corrcoef(price_window, volume_window)[0, 1]
        correlations.iloc[i] = corr if not np.isnan(corr) else 0
    
    # Calculate divergence factor
    for i in range(len(data)):
        if pd.notna(price_slopes.iloc[i]) and pd.notna(correlations.iloc[i]):
            factor.iloc[i] = price_slopes.iloc[i] * (1 - correlations.iloc[i])
    
    return factor
