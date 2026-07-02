import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Volume Trend Component
    volume = df['volume']
    volume_slopes = []
    for t in range(len(volume)):
        if t < 4:
            volume_slopes.append(np.nan)
        else:
            X = np.arange(5).reshape(-1, 1)
            y = volume[t-4:t+1].values
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            avg_volume = np.mean(y)
            normalized_slope = slope / avg_volume
            volume_slopes.append(normalized_slope)
    volume_slopes = pd.Series(volume_slopes, index=df.index)
    
    # Volume Direction
    volume_direction = np.sign(volume_slopes)
    
    # Price Trend Component
    close = df['close']
    price_slopes = []
    for t in range(len(close)):
        if t < 4:
            price_slopes.append(np.nan)
        else:
            X = np.arange(5).reshape(-1, 1)
            y = close[t-4:t+1].values
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            avg_price = np.mean(y)
            normalized_slope = slope / avg_price
            price_slopes.append(normalized_slope)
    price_slopes = pd.Series(price_slopes, index=df.index)
    
    # Price Direction
    price_direction = np.sign(price_slopes)
    
    # Divergence Signal
    direction_signal = volume_direction * price_direction
    magnitude_adjustment = np.abs(volume_slopes - price_slopes)
    divergence_signal = direction_signal * magnitude_adjustment
    
    return divergence_signal
