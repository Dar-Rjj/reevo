import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate 5-day Price Slope
    def calculate_price_slope(series):
        X = np.arange(len(series)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, series)
        return model.coef_[0]
    
    price_slope = df['close'].rolling(window=5).apply(calculate_price_slope, raw=True)
    
    # Calculate Intraday Strength
    intraday_strength = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Calculate 5-day Volume Slope
    volume_slope = df['volume'].rolling(window=5).apply(calculate_price_slope, raw=True)
    
    # Combine Signals
    factor = price_slope * volume_slope * intraday_strength
    
    return factor
