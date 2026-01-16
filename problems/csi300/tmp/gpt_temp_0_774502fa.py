import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Price Trend Component
    # Rolling 5-day price slope
    def rolling_slope(series, window):
        slopes = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            x = np.arange(window).reshape(-1, 1)
            y = series[i - window + 1:i + 1].values
            model = LinearRegression().fit(x, y)
            slopes[i] = model.coef_[0]
        return slopes
    
    df['price_slope'] = rolling_slope(df['close'], 5)
    
    # Normalize to range [-1, 1]
    def normalize_to_range(series, target_min, target_max):
        min_val = series.min()
        max_val = series.max()
        normalized = (series - min_val) / (max_val - min_val)
        return normalized * (target_max - target_min) + target_min
    
    df['price_trend'] = normalize_to_range(df['price_slope'].rolling(window=5, min_periods=1).mean(), -1, 1)
    
    # Liquidity Momentum
    # Volume-Weighted Price Change
    rolling_mean_volume = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_weighted_price_change'] = (df['close'] - df['open']) * df['volume'] / rolling_mean_volume
    
    # Normalize to range [0, 1]
    df['liquidity_momentum'] = normalize_to_range(df['volume_weighted_price_change'].rolling(window=10, min_periods=1).mean(), 0, 1)
    
    # Adjusted Momentum
    # Multiply Price Trend by Liquidity Momentum
    df['adjusted_momentum'] = df['price_trend'] * df['liquidity_momentum']
    
    # Rolling 5-day smoothing
    df['factor'] = df['adjusted_momentum'].rolling(window=5, min_periods=1).mean()
    
    return df['factor']
