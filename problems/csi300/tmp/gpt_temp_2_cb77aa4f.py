import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate price trend component
    price_slopes = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data['close'].iloc[i-4:i+1]
        if len(window) < 2:
            continue
        X = np.arange(len(window)).reshape(-1, 1)
        y = window.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        price_slopes.iloc[i] = model.coef_[0][0] / data['close'].iloc[i]
    
    # Calculate volume trend component
    volume_slopes = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data['volume'].iloc[i-4:i+1]
        if len(window) < 2:
            continue
        X = np.arange(len(window)).reshape(-1, 1)
        y = window.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        volume_slopes.iloc[i] = model.coef_[0][0] / data['volume'].iloc[i]
    
    # Calculate divergence signal
    for i in range(len(data)):
        if pd.isna(price_slopes.iloc[i]) or pd.isna(volume_slopes.iloc[i]):
            continue
        
        price_slope = price_slopes.iloc[i]
        volume_slope = volume_slopes.iloc[i]
        
        # Check for divergence conditions
        if (price_slope > 0 and volume_slope < 0) or (price_slope < 0 and volume_slope > 0):
            denominator = abs(price_slope) + abs(volume_slope)
            if denominator != 0:
                factor.iloc[i] = (price_slope - volume_slope) / denominator
            else:
                factor.iloc[i] = 0
        else:
            factor.iloc[i] = 0
    
    return factor
