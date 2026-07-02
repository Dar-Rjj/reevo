import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Measure Intraday Momentum
    price_change = df['close'] - df['open']
    normalized_momentum = price_change / df['open']
    
    # Adjust for Volume Intensity
    rolling_avg_volume = df['volume'].rolling(window=5).mean()
    volume_ratio = df['volume'] / rolling_avg_volume
    
    # Detect Divergence from Trend
    def calculate_slope(series):
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0][0]
    
    price_trend = df['close'].rolling(window=5).apply(calculate_slope)
    momentum_divergence = normalized_momentum - price_trend
    
    # Intraday Volume-Weighted Momentum Divergence
    factor = momentum_divergence * volume_ratio
    return factor
