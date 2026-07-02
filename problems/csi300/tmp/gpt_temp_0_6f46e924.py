import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate 5-day rolling price range
    df['price_range'] = df['high'] - df['low']
    df['rolling_range'] = df['price_range'].rolling(window=5).mean()
    
    # Compute range consistency
    df['rolling_range_std'] = df['price_range'].rolling(window=5).std()
    df['range_consistency'] = df['rolling_range_std'] / df['rolling_range']
    
    # Calculate volume slope using linear regression
    def calculate_slope(series):
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0][0]
    
    df['volume_slope'] = df['volume'].rolling(window=5).apply(calculate_slope, raw=False)
    
    # Adjust range consistency by volume slope
    df['factor'] = df['range_consistency'] * df['volume_slope']
    
    return df['factor']
