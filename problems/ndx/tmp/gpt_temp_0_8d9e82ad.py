import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Calculate price change (close - open)
    price_change = data['close'] - data['open']
    
    # Calculate intraday range (high - low)
    intraday_range = data['high'] - data['low']
    
    # Calculate price elasticity (price change normalized by intraday range)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    price_elasticity = price_change / (intraday_range + epsilon)
    
    # Calculate volume trend (5-day rolling slope)
    volume_trend = pd.Series(index=data.index, dtype=float)
    window = 5
    for i in range(len(data)):
        if i >= window - 1:
            # Use only past and current data (no future)
            window_data = data['volume'].iloc[i-window+1:i+1]
            x = np.arange(len(window_data))
            slope = linregress(x, window_data)[0]
            volume_trend.iloc[i] = slope
    
    # Fill initial NaN values with 0
    volume_trend = volume_trend.fillna(0)
    
    # Combine factors: price elasticity * volume trend
    factor = price_elasticity * volume_trend
    
    return factor
