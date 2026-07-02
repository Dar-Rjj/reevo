import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Initialize output Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Price Trend Component
    close_prices = data['close']
    price_slopes = pd.Series(index=data.index, dtype=float)
    
    # Volume Trend Component
    volumes = data['volume']
    volume_slopes = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling slopes for price and volume
    for i in range(4, len(data)):
        # Price slope calculation (5-day window)
        price_window = close_prices.iloc[i-4:i+1]
        price_slope = linregress(np.arange(5), price_window.values).slope
        price_slopes.iloc[i] = price_slope / close_prices.iloc[i]  # Normalized by price level
        
        # Volume slope calculation (5-day window)
        volume_window = volumes.iloc[i-4:i+1]
        volume_slope = linregress(np.arange(5), volume_window.values).slope
        # Normalized by 20-day average volume
        avg_volume = volumes.iloc[max(0, i-19):i+1].mean()
        volume_slopes.iloc[i] = volume_slope / avg_volume if avg_volume != 0 else 0
    
    # Calculate divergence factor
    for i in range(4, len(data)):
        if not np.isnan(price_slopes.iloc[i]) and not np.isnan(volume_slopes.iloc[i]):
            abs_diff = abs(price_slopes.iloc[i] - volume_slopes.iloc[i])
            sign_adjust = np.sign(price_slopes.iloc[i] * volume_slopes.iloc[i])
            factor.iloc[i] = abs_diff * sign_adjust
    
    return factor
