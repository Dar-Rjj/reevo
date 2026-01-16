import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Compute Price Momentum
    price_momentum = (data['close'] - data['open']) / data['open']
    
    # Calculate 5-day Volume Slope
    def volume_slope(series):
        return linregress(np.arange(5), series.values).slope
    
    volume_slope_series = data['volume'].rolling(window=5).apply(volume_slope, raw=False)
    
    # Adjust Price Momentum by Volume Slope
    adjusted_momentum = price_momentum * volume_slope_series
    
    # Calculate 10-day standard deviation of returns
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=10).std()
    
    # Scale Adjusted Momentum by Volatility
    factor = adjusted_momentum / volatility
    
    return factor
