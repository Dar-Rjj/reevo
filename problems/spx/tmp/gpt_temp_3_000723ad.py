import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()
    
    # Function to calculate the slope of a 5-day linear regression
    def calculate_slope(series):
        if len(series) < 5:
            return np.nan
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    # Calculate the short-term price trend (5-day slope of close prices)
    price_trend = df['close'].rolling(window=5, min_periods=5).apply(calculate_slope, raw=False)
    
    # Calculate the volume trend (5-day slope of volume)
    volume_trend = df['volume'].rolling(window=5, min_periods=5).apply(calculate_slope, raw=False)
    
    # Compute divergence by multiplying price slope by volume slope and taking the sign
    divergence = np.sign(price_trend * volume_trend)
    
    return divergence
