import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(data):
    # Price Trend Component
    price_trend = data['close'] / data['close'].shift(5)
    price_trend_smooth = price_trend.ewm(span=5, adjust=False).mean()
    
    # Volume Trend Component
    volume_trend = data['volume'] / data['volume'].shift(5)
    volume_trend_smooth = volume_trend.ewm(span=5, adjust=False).mean()
    
    # Divergence Signal
    divergence = price_trend_smooth - volume_trend_smooth
    
    # Normalize Divergence
    z_score = divergence.rolling(window=20).apply(lambda x: zscore(x)[-1], raw=True)
    normalized_divergence = 1 / (1 + np.exp(-z_score))
    
    return normalized_divergence
