import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Price Divergence Component
    # Calculate 5-day rolling mean of Close price (using only past data)
    close_mean = data['close'].rolling(window=5, min_periods=1).mean()
    
    # Calculate price deviation from mean
    price_deviation = data['close'] - close_mean
    
    # Normalize by price range (High - Low)
    price_range = data['high'] - data['low']
    normalized_divergence = price_deviation / (price_range + 1e-6)  # Add small constant to avoid division by zero
    
    # Volume Confirmation
    # Calculate 5-day rolling slope of Volume
    def rolling_slope(series, window=5):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i >= window - 1:
                x = np.arange(window)
                y = series.iloc[i-window+1:i+1].values
                slope = np.polyfit(x, y, 1)[0]
                slopes.iloc[i] = slope
            else:
                slopes.iloc[i] = np.nan
        return slopes.ffill()  # Forward fill for initial periods
    
    volume_slope = rolling_slope(data['volume'], window=5)
    
    # Calculate volume trend (current volume * slope direction)
    volume_trend = data['volume'] * np.sign(volume_slope)
    
    # Weight divergence by volume trend and scale by price range
    factor = normalized_divergence * volume_trend / (price_range + 1e-6)
    
    return factor
