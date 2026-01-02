import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate High-Low Range (Intraday Volatility)
    df['high_low_range'] = df['high'] - df['low']
    
    # Compute Volume Trend Strength
    # Prepare volume data for linear regression
    volume = df['volume'].values
    X = np.arange(5).reshape(-1, 1)  # Independent variable for regression
    
    # Initialize volume slopes
    volume_slopes = np.zeros(len(df))
    
    for i in range(4, len(df)):
        y = volume[i-4:i+1]  # Use current and past 4 days data
        model = LinearRegression()
        model.fit(X, y)
        volume_slopes[i] = model.coef_[0]  # Slope of the regression line
    
    # Calculate absolute trend strength
    df['volume_trend_strength'] = np.abs(volume_slopes)
    
    # Avoid division by zero by replacing zero with a small value
    df['volume_trend_strength'].replace(0, 1e-10, inplace=True)
    
    # Adjust Intraday Volatility by Volume Trend Strength
    df['adjusted_volatility'] = df['high_low_range'] / df['volume_trend_strength']
    
    return df['adjusted_volatility']
