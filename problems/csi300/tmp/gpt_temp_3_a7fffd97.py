import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Trend Component: 5-day Price Slope
    def calculate_slope(series, window):
        slopes = np.zeros(len(series))
        for i in range(window - 1, len(series)):
            y = series.iloc[i - window + 1:i + 1].values
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            slopes[i] = slope
        return pd.Series(slopes, index=series.index)

    price_slope = calculate_slope(df['close'], 5)
    
    # Volume Trend Component: 5-day Volume Slope
    volume_slope = calculate_slope(df['volume'], 5)
    
    # Formulate Divergence Signal
    divergence_signal = price_slope / volume_slope.replace(0, np.nan)
    
    # Apply Z-Score Normalization
    z_score = (divergence_signal - divergence_signal.rolling(window=20, min_periods=1).mean()) / divergence_signal.rolling(window=20, min_periods=1).std()
    
    return z_score
