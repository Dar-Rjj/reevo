import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate 5-day rolling slope using linear regression
    def rolling_slope(series):
        if len(series) < 5:
            return np.nan
        X = np.array(range(5)).reshape(-1, 1)
        y = series.values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    # Calculate sign consistency
    def sign_consistency(slopes):
        if len(slopes) < 5:
            return np.nan
        signs = np.sign(slopes)
        return np.mean(signs == signs[-1])
    
    # Calculate normalized volatility
    def normalized_volatility(df):
        daily_range = df['high'] - df['low']
        avg_range = daily_range.rolling(window=20, min_periods=1).mean()
        return daily_range / avg_range
    
    # Calculate 5-day rolling slope for close prices
    slopes = df['close'].rolling(window=5, min_periods=5).apply(rolling_slope, raw=False)
    
    # Calculate sign consistency over the same window
    sign_consistencies = slopes.rolling(window=5, min_periods=5).apply(sign_consistency, raw=False)
    
    # Calculate normalized volatility
    vol_ratio = normalized_volatility(df)
    
    # Combine trend persistence with volatility adjustment
    factor = sign_consistencies * vol_ratio
    
    # Cap extreme values
    factor = np.clip(factor, -2, 2)
    
    return factor
