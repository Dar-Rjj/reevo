import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # 1. Measure Intraday Price Acceleration
    # Calculate Price Change Rate
    numerator = df['close'] - df['open']
    denominator = df['high'] - df['low']
    # Avoid division by zero
    denominator = denominator.replace(0, np.nan)
    price_acceleration = numerator / denominator
    
    # 2. Adjust by Volume Trend
    # Calculate Volume Slope using rolling linear regression
    def rolling_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < window - 1:
                slopes.iloc[i] = np.nan
                continue
            window_data = series.iloc[i - window + 1:i + 1]
            x = np.arange(len(window_data))
            y = window_data.values
            # Simple linear regression slope calculation
            slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum()
            slopes.iloc[i] = slope
        return slopes
    
    volume_slope = rolling_slope(df['volume'], window=5)
    
    # Combine signals
    factor = price_acceleration * volume_slope
    
    # Return as a Series
    return factor
