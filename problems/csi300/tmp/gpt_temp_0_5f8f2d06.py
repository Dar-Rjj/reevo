import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Trend Component
    def linear_regression_slope(series, window):
        slopes = np.zeros(len(series))
        for i in range(window-1, len(series)):
            y = series[i-window+1:i+1].values
            x = np.arange(window)
            slope = linregress(x, y)[0]
            slopes[i] = slope
        return pd.Series(slopes, index=series.index)
    
    short_term_trend = linear_regression_slope(df['close'], 5)
    long_term_trend = linear_regression_slope(df['close'], 20)
    
    # Volume Trend Component
    volume_roc = df['volume'].pct_change(5)
    volume_std = df['volume'].rolling(20).std()
    normalized_volume = volume_roc / volume_std
    
    # Z-score scaling
    mean_vol = normalized_volume.rolling(20).mean()
    std_vol = normalized_volume.rolling(20).std()
    zscore_volume = (normalized_volume - mean_vol) / std_vol
    
    # Divergence Signal
    price_up = short_term_trend > 0
    price_down = short_term_trend < 0
    volume_up = zscore_volume > 0
    volume_down = zscore_volume < 0
    
    signal = np.zeros(len(df))
    signal[(price_up & volume_down)] = -1  # Negative signal
    signal[(price_down & volume_up)] = 1   # Positive signal
    
    # Weight by Trend Strength
    weighted_signal = signal * np.abs(short_term_trend)
    
    # Logistic Transformation
    factor = 1 / (1 + np.exp(-weighted_signal))
    
    return pd.Series(factor, index=df.index)
