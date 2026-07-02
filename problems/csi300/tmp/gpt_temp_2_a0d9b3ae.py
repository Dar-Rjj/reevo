import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Volume Trend Analysis
    # Compute 5-day Volume Slope using linear regression
    volume = df['volume'].values
    dates = np.arange(len(volume)).reshape(-1, 1)
    volume_slope = np.zeros_like(volume, dtype=float)
    for i in range(4, len(volume)):
        model = LinearRegression()
        model.fit(dates[i-4:i+1], volume[i-4:i+1])
        volume_slope[i] = model.coef_[0]
    
    # Compute 5-day Volume MA
    volume_ma = df['volume'].rolling(window=5, min_periods=1).mean()
    
    # Volume Ratio: Current Volume / 5-day Volume MA
    volume_ratio = df['volume'] / volume_ma
    
    # Price Momentum Divergence
    # Compute 3-day Price Return
    price_return = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Compare Price Return Sign with Volume Trend Sign
    momentum_direction = np.where(
        np.sign(price_return) != np.sign(volume_slope),
        -1, 1
    )
    
    # Signal Combination
    factor = volume_ratio * momentum_direction
    
    return factor
