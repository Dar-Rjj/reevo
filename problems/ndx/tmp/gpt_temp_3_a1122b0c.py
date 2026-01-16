import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Compute Intraday Return
    intraday_return = (df['high'] - df['low']) / df['open']
    intraday_return *= np.sign(df['close'] - df['open'])
    
    # Calculate 5-day Rolling Volume Slope
    def rolling_slope(series, window=5):
        slopes = np.zeros(len(series))
        for i in range(window, len(series)):
            x = np.arange(window).reshape(-1, 1)
            y = series[i-window:i].values
            model = LinearRegression().fit(x, y)
            slopes[i] = model.coef_[0]
        return slopes
    
    volume_slope = rolling_slope(df['volume'], window=5)
    
    # Adjust Momentum by Volume Trend
    momentum_adjusted = intraday_return * volume_slope
    
    # Calculate Volume Z-Score
    rolling_std = df['volume'].rolling(window=20, min_periods=1).std()
    volume_zscore = df['volume'] / rolling_std
    
    # Combine Signals
    combined_signal = momentum_adjusted * volume_zscore
    
    # Apply Min-Max Normalization
    min_val = combined_signal.rolling(window=20, min_periods=1).min()
    max_val = combined_signal.rolling(window=20, min_periods=1).max()
    normalized_signal = (combined_signal - min_val) / (max_val - min_val)
    
    return normalized_signal
