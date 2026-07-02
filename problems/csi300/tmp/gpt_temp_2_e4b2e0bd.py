import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Calculate daily range (current day only)
    daily_range = data['high'] - data['low']
    
    # Calculate 5-day price trend (using past 5 days)
    trend_slopes = pd.Series(np.nan, index=data.index)
    for i in range(5, len(data)):
        window = data['close'].iloc[i-5:i]
        slope = linregress(np.arange(5), window.values).slope
        trend_slopes.iloc[i] = slope
    
    # Scale range by trend strength (absolute value)
    adjusted_range = daily_range * trend_slopes.abs()
    
    # Calculate 20-day average range (using past 20 days)
    avg_range = daily_range.rolling(window=20, min_periods=1).mean()
    
    # Normalize adjusted range by average range
    factor = adjusted_range / avg_range
    
    # Handle any potential NaN values (first few days)
    factor = factor.fillna(0)
    
    return factor
