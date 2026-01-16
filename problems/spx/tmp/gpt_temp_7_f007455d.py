import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate Price Range
    price_range = df['high'] - df['low']
    
    # Compute Volume Slope using Linear Regression over 5 days
    def compute_volume_slope(volume_series):
        X = np.arange(len(volume_series)).reshape(-1, 1)
        y = volume_series.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    volume_slope = df['volume'].rolling(window=5).apply(compute_volume_slope, raw=False)
    
    # Incorporate Volume Momentum by multiplying Price Range by Volume Slope
    adjusted_range = price_range * volume_slope
    
    # Calculate Rolling Mean Price Range over 10 days
    rolling_mean_price_range = price_range.rolling(window=10).mean()
    
    # Normalize Adjusted Range by Rolling Mean Price Range
    normalized_range = adjusted_range / rolling_mean_price_range
    
    return normalized_range
