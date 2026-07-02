import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress, zscore

def heuristics_v2(df):
    # Calculate Price Trend (5-day Linear Regression Slope of Close prices)
    def price_trend(close):
        return close.rolling(window=5).apply(lambda x: linregress(np.arange(5), x).slope, raw=True)
    
    # Calculate Volume Trend (5-day Linear Regression Slope of log(Volume))
    def volume_trend(volume):
        log_volume = np.log(volume)
        return log_volume.rolling(window=5).apply(lambda x: linregress(np.arange(5), x).slope, raw=True)
    
    # Calculate Price-Volume Divergence
    price_divergence = price_trend(df['close'])
    volume_divergence = volume_trend(df['volume'])
    divergence = price_divergence - volume_divergence
    
    # Calculate Normalized Range
    normalized_range = (df['high'] - df['low']) / df['close']
    
    # Combine Signals
    combined_signal = divergence * normalized_range
    
    # Apply Rolling Z-Score (5-day)
    rolling_zscore = combined_signal.rolling(window=5).apply(lambda x: zscore(x)[-1], raw=True)
    
    return rolling_zscore
