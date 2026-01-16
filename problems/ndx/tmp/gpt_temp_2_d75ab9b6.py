import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate 5-day Price Slope using Close prices
    def calculate_price_slope(close_prices):
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    # Calculate 5-day Volume Slope using Volume
    def calculate_volume_slope(volumes):
        X = np.arange(len(volumes)).reshape(-1, 1)
        y = volumes.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    # Compute rolling 5-day Price Slope
    df['price_slope'] = df['close'].rolling(window=5).apply(calculate_price_slope, raw=False)
    
    # Compute rolling 5-day Volume Slope
    df['volume_slope'] = df['volume'].rolling(window=5).apply(calculate_volume_slope, raw=False)
    
    # Compute Divergence: Price Slope - Volume Slope
    df['divergence'] = df['price_slope'] - df['volume_slope']
    
    # Normalize by Price Volatility: Rolling 5-day std of Close returns
    df['close_returns'] = df['close'].pct_change()
    df['price_volatility'] = df['close_returns'].rolling(window=5).std()
    
    # Avoid division by zero
    df['price_volatility'] = df['price_volatility'].replace(0, np.nan)
    
    # Normalize Divergence by Price Volatility
    df['price_volume_divergence_factor'] = df['divergence'] / df['price_volatility']
    
    # Return the factor as a Series
    return pd.Series(df['price_volume_divergence_factor'], index=df.index)
