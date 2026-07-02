import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Calculate 5-day linear regression slope for price and volume
    def linear_regression_slope(series, window):
        return series.rolling(window=window).apply(lambda x: np.polyfit(np.arange(window), x, 1)[0], raw=True)
    
    # Calculate price trend
    price_trend = linear_regression_slope(data['close'], 5)
    
    # Calculate volume trend
    volume_trend = linear_regression_slope(data['volume'], 5)
    
    # Normalize price trend
    price_trend_norm = price_trend / data['close'].shift(5) * 100
    
    # Normalize volume trend
    volume_trend_norm = volume_trend / data['volume'].shift(5) * 100
    
    # Generate signals
    positive_divergence = (price_trend_norm > 0.5) & (volume_trend_norm < -1)
    negative_divergence = (price_trend_norm < -0.5) & (volume_trend_norm > 1)
    
    # Combine signals
    signal = pd.Series(np.where(positive_divergence, 1, np.where(negative_divergence, -1, 0)), index=data.index)
    
    return signal
