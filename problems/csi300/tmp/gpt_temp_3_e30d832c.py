import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate 5-day momentum of Close price
    momentum_5 = df['close'].diff(5)
    
    # Calculate 10-day momentum of Close price
    momentum_10 = df['close'].diff(10)
    
    # Compute Second Derivative of Price (Price Acceleration)
    price_acceleration = momentum_5 - momentum_10
    
    # Normalize Price Acceleration by Price Level
    price_acceleration_normalized = (price_acceleration / df['close']) * 100
    
    # Calculate Volume Slope (Trend)
    volume_slope = df['volume'].rolling(window=5).apply(
        lambda x: LinearRegression().fit(np.arange(5).reshape(-1, 1), x).coef_[0], raw=True
    )
    
    # Normalize Volume Slope by average Volume
    volume_trend = volume_slope / df['volume'].rolling(window=5).mean()
    
    # Apply Volume Weighting to Price Acceleration
    factor = price_acceleration_normalized * volume_trend
    
    return factor
