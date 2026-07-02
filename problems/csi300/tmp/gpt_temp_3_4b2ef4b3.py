import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(data):
    # Calculate Recent Price Change
    data['price_change'] = data['close'].diff()
    
    # Normalize by Price Level
    data['normalized_price_change'] = data['price_change'] / data['open']
    
    # Calculate Volume Trend (5-Day Rolling Volume Slope)
    def rolling_slope(series):
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    data['volume_trend'] = data['volume'].rolling(window=5).apply(rolling_slope, raw=False)
    
    # Adjust by Volume Trend
    data['factor'] = data['normalized_price_change'] * data['volume_trend']
    
    # Return the factor series
    return data['factor']
