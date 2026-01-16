import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Calculate rolling 5-day slope (trend) using linear regression
    def rolling_slope(series, window):
        slopes = np.zeros(len(series))
        for i in range(window, len(series)):
            y = series.iloc[i-window:i]
            x = np.arange(window)
            slope = linregress(x, y)[0]
            slopes[i] = slope
        return pd.Series(slopes, index=series.index)
    
    close_slope = rolling_slope(df['close'], 5)
    
    # Calculate rolling 5-day price range (high - low)
    rolling_high = df['high'].rolling(5, min_periods=1).max()
    rolling_low = df['low'].rolling(5, min_periods=1).min()
    price_range = rolling_high - rolling_low
    
    # Normalize trend by price range (avoid division by zero)
    normalized_trend = close_slope / (price_range.replace(0, np.nan))
    
    # Calculate recent price change (close(t)/close(t-5) - 1)
    recent_change = df['close'] / df['close'].shift(5) - 1
    
    # Combine components to get final factor
    factor = normalized_trend * recent_change
    
    return factor
