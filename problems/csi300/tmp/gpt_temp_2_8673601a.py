import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def heuristics_v2(df):
    # Compute Price Trend
    def calculate_slope(values):
        X = np.arange(len(values)).reshape(-1, 1)
        y = values.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        return reg.coef_[0][0]
    
    df['price_slope_5'] = df['close'].rolling(window=5).apply(calculate_slope, raw=False)
    df['price_slope_20'] = df['close'].rolling(window=20).apply(calculate_slope, raw=False)
    
    # Compute Volume Trend
    df['volume_slope_5'] = df['volume'].rolling(window=5).apply(calculate_slope, raw=False)
    df['volume_slope_20'] = df['volume'].rolling(window=20).apply(calculate_slope, raw=False)
    
    # Combine Trend Components
    df['combined_5'] = df['price_slope_5'] * df['volume_slope_5']
    df['combined_20'] = df['price_slope_20'] * df['volume_slope_20']
    
    # Final Factor
    factor = df['combined_5'] / df['combined_20']
    factor = (factor - factor.mean()) / factor.std()
    
    return factor
