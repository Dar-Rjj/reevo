import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Rolling regression slope of price against volume
    def rolling_regression_slope(x, y, window):
        slopes = np.zeros(len(x))
        for i in range(window, len(x)):
            model = LinearRegression().fit(x[i-window:i].reshape(-1, 1), y[i-window:i])
            slopes[i] = model.coef_[0]
        return slopes

    window = 10
    slopes = rolling_regression_slope(df['volume'].values.reshape(-1, 1), df['close'].values, window)
    
    # Volatility-scaled price changes
    volatility_scaled_price_changes = (df['close'].diff() / df['close'].rolling(window).std())
    
    # Cumulative volume-adjusted returns
    volume_adjusted_returns = (df['close'].pct_change() * df['volume'].shift(1)).cumsum()
    
    # Combine components
    heuristics_matrix = pd.Series(
        slopes * volatility_scaled_price_changes + volume_adjusted_returns,
        index=df.index
    )
    
    return heuristics_matrix
