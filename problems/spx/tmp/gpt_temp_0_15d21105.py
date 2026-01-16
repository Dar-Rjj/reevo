import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate 3-day return
    df['3_day_return'] = df['close'] / df['close'].shift(3) - 1
    
    # Invert recent performance
    df['reversal'] = -df['3_day_return']
    
    # Calculate volume slope using linear regression
    def volume_slope(series):
        model = LinearRegression()
        X = np.arange(len(series)).reshape(-1, 1)
        model.fit(X, series)
        return model.coef_[0]
    
    df['volume_slope'] = df['volume'].rolling(window=5).apply(volume_slope, raw=True)
    
    # Scale by price range
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # Combine signals
    df['factor'] = df['reversal'] * df['volume_slope'] * df['price_range']
    
    return df['factor']
