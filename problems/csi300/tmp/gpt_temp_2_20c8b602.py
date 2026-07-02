import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Trend Component
    def calculate_price_slope(close):
        slopes = []
        for i in range(len(close)):
            if i >= 4:
                x = np.arange(5)
                y = close.iloc[i-4:i+1].values
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=close.index)
    
    # Volume Trend Component
    def calculate_volume_slope(volume):
        slopes = []
        for i in range(len(volume)):
            if i >= 4:
                x = np.arange(5)
                y = volume.iloc[i-4:i+1].values
                slope, _, _, _, _ = linregress(x, y)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=volume.index)
    
    # Normalize Price Slope by Price Volatility
    price_slope = calculate_price_slope(df['close'])
    price_vol = df['close'].rolling(window=5).std()
    normalized_price_slope = price_slope / price_vol
    
    # Volume Divergence Component
    volume_slope = calculate_volume_slope(df['volume'])
    divergence_signal = np.sign(price_slope) * volume_slope
    volume_intensity = df['volume'] / df['volume'].rolling(window=5).mean()
    final_divergence = divergence_signal * volume_intensity
    
    # Combine Price Trend and Volume Divergence
    factor = normalized_price_slope * final_divergence
    return factor
