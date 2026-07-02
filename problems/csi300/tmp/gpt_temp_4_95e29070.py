import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Compute Closing Price Change
    price_change = df['close'].diff()
    
    # Normalize Price Change using Prior 10-day StdDev of Returns
    returns = df['close'].pct_change()
    std_price = returns.rolling(window=10).std()
    normalized_price_change = price_change / std_price
    
    # Compute Volume Trend Strength using 5-day Volume Slope
    volume_slope = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:  # Need at least 5 days for regression
            X = np.arange(5).reshape(-1, 1)
            y = df['volume'].iloc[i-4:i+1].values
            model = LinearRegression().fit(X, y)
            volume_slope.iloc[i] = model.coef_[0]
    
    # Normalize Volume Slope using Prior 5-day StdDev of Volume
    std_volume = df['volume'].rolling(window=5).std()
    normalized_volume_slope = volume_slope / std_volume
    
    # Combine Signals
    factor = normalized_price_change * normalized_volume_slope
    
    return factor
