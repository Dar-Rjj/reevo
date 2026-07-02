import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Measure Price Sensitivity: Rolling Price Variability
    price_variability = df['high'] - df['low']
    rolling_price_variability = price_variability.rolling(window=10).mean()
    
    # Normalize Price Variability
    normalized_price_variability = rolling_price_variability / df['close'].shift(1)
    scaled_price_variability = np.log1p(normalized_price_variability)
    
    # Quantify Volume Trend: Volume Slope
    def rolling_volume_slope(series):
        if len(series) < 10:
            return np.nan
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    volume_slope = df['volume'].rolling(window=10).apply(rolling_volume_slope, raw=False)
    
    # Normalize Volume Slope
    rolling_volume_avg = df['volume'].rolling(window=10).mean()
    normalized_volume_slope = volume_slope / rolling_volume_avg
    scaled_volume_slope = np.log1p(normalized_volume_slope)
    
    # Combine Signals: Adjust Price Sensitivity by Volume Trend
    combined_signal = scaled_price_variability * scaled_volume_slope
    combined_signal_scaled = np.log1p(combined_signal)
    
    # Directional Adjustment
    direction = np.where(volume_slope > 0, 1, -1)
    final_signal = combined_signal_scaled * direction
    
    return final_signal
