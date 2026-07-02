import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate Price Momentum
    price_momentum = df['close'] - df['close'].shift(5)
    
    # Calculate Volume Slope using Linear Regression
    volume_slopes = []
    for t in range(len(df)):
        if t < 5:
            volume_slopes.append(np.nan)
            continue
        X = np.arange(5).reshape(-1, 1)
        y = df['volume'].iloc[t-5:t].values
        reg = LinearRegression().fit(X, y)
        volume_slopes.append(reg.coef_[0])
    
    df['volume_slope'] = volume_slopes
    
    # Normalize Momentum by Volume Slope
    factor = price_momentum / df['volume_slope']
    
    # Ensure positive Volume Slope
    factor = factor.mask(df['volume_slope'] <= 0, np.nan)
    
    return factor
