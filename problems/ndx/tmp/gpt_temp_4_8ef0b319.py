import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Price Trend Component
    # Calculate 5-day Price Trend using Close price and Linear regression slope
    def calculate_trend(data, window):
        trend = np.zeros(len(data))
        for i in range(window-1, len(data)):
            X = np.arange(window).reshape(-1, 1)
            y = data.iloc[i-window+1:i+1].values
            model = LinearRegression().fit(X, y)
            trend[i] = model.coef_[0]
        return pd.Series(trend, index=data.index)
    
    price_trend = calculate_trend(df['close'], 5)
    
    # Normalize Price Trend by Volatility
    returns = df['close'].pct_change()
    volatility = returns.rolling(window=20, min_periods=1).std()
    normalized_price_trend = price_trend / volatility
    
    # Volume Trend Component
    # Calculate 5-day Volume Trend using Volume and Linear regression slope
    volume_trend = calculate_trend(df['volume'], 5)
    
    # Normalize Volume Trend by Average Volume
    avg_volume = df['volume'].rolling(window=20, min_periods=1).mean()
    normalized_volume_trend = volume_trend / avg_volume
    
    # Price-Volume Divergence Factor
    divergence_factor = normalized_price_trend * normalized_volume_trend
    
    return divergence_factor
