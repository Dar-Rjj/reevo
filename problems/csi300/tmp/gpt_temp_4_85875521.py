import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(data):
    # Compute Momentum Ratio
    momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Calculate Volume Slope
    def calculate_slope(volume):
        X = np.arange(len(volume)).reshape(-1, 1)
        model = LinearRegression().fit(X, volume)
        return model.coef_[0]
    
    volume_slope = data['volume'].rolling(window=5).apply(calculate_slope, raw=False)
    
    # Combine Momentum and Volume Trend
    momentum_volume = momentum * volume_slope
    
    # Normalize by Historical Momentum Volatility
    momentum_volatility = momentum.rolling(window=10).std()
    factor = momentum_volume / momentum_volatility
    
    # Return the factor as a pandas Series
    return factor
