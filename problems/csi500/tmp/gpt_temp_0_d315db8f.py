import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Momentum Component
    # Calculate 3-day return using High price
    df['high_return'] = df['high'].pct_change(periods=3)
    
    # Normalize by Volatility: 10-day rolling std of returns
    df['volatility'] = df['high'].pct_change().rolling(window=10).std()
    df['momentum'] = df['high_return'] / df['volatility']
    
    # Volume Component
    # Calculate Volume Slope using 3-day linear regression
    def calculate_volume_slope(series):
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0][0]
    
    df['volume_slope'] = df['amount'].rolling(window=3).apply(calculate_volume_slope)
    
    # Combine Components
    # Multiply Momentum by Volume Slope
    df['combined'] = df['momentum'] * df['volume_slope']
    
    # Normalize by Average Momentum Ratio (past 3 days)
    df['normalized_combined'] = df['combined'] / df['combined'].rolling(window=3).mean()
    
    return df['normalized_combined']
