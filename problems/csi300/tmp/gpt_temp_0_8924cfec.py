import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Calculate intraday mid-price
    df['mid_price'] = (df['high'] + df['low']) / 2
    
    # Calculate intraday price slope over last 5 minutes
    def calculate_slope(window):
        X = np.arange(len(window)).reshape(-1, 1)
        y = window.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    df['intraday_slope'] = df['mid_price'].rolling(window=5).apply(calculate_slope, raw=False)
    
    # Calculate 5-day average of intraday slopes
    df['slope_5d_avg'] = df['intraday_slope'].rolling(window=5*24*60).mean()
    
    # Calculate intraday momentum acceleration
    df['momentum_acc'] = df['intraday_slope'] - df['slope_5d_avg']
    
    # Normalize momentum acceleration by 20-day standard deviation of returns
    df['returns'] = df['close'].pct_change()
    df['volatility_20d'] = df['returns'].rolling(window=20*24*60).std()
    df['norm_momentum_acc'] = df['momentum_acc'] / df['volatility_20d']
    
    # Apply volume confirmation
    df['volume_10d_avg'] = df['volume'].rolling(window=10*24*60).mean()
    df['volume_confirmation'] = df['volume'] / df['volume_10d_avg']
    
    # Combine normalized momentum acceleration with volume confirmation
    df['factor'] = df['norm_momentum_acc'] * df['volume_confirmation']
    
    return df['factor']
