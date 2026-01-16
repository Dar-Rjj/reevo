import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Calculate Rolling Mean of Spread
    df['rolling_mean_spread'] = df['high_low_spread'].rolling(window=10, min_periods=1).mean()
    
    # Normalize Spread by Rolling Mean
    df['normalized_spread'] = df['high_low_spread'] / df['rolling_mean_spread']
    
    # Calculate Trend Strength
    def calculate_slope(window):
        X = np.arange(len(window)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, window)
        return model.coef_[0]
    
    df['close_slope'] = df['close'].rolling(window=10, min_periods=1).apply(calculate_slope, raw=True)
    df['mean_abs_slope'] = df['close_slope'].abs().rolling(window=10, min_periods=1).mean()
    df['trend_strength'] = df['close_slope'] / df['mean_abs_slope']
    
    # Multiply by High-Low Spread Momentum
    factor = df['normalized_spread'] * df['trend_strength']
    
    return factor.dropna()
