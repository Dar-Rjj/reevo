import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Price-Volume Divergence calculation
    for i in range(4, len(df)):
        # Price Trend Component - 5-day slope
        price_window = df['close'].iloc[i-4:i+1]
        x = np.arange(5)
        price_slope = np.polyfit(x, price_window, 1)[0]
        
        # Volume Trend Component - 5-day slope
        volume_window = df['volume'].iloc[i-4:i+1]
        volume_slope = np.polyfit(x, volume_window, 1)[0]
        
        # Compute Divergence
        divergence = np.sign(price_slope * volume_slope)
        
        # Range Confirmation
        current_range = (df['high'].iloc[i] - df['low'].iloc[i]) / df['close'].iloc[i]
        
        # Historical Range - median of past 10 days
        if i >= 9:
            historical_ranges = [
                (df['high'].iloc[j] - df['low'].iloc[j]) / df['close'].iloc[j]
                for j in range(i-9, i+1)
            ]
            median_range = np.median(historical_ranges)
            range_ratio = current_range / median_range
        else:
            range_ratio = 1.0  # neutral value when not enough history
        
        # Combine components
        factor.iloc[i] = divergence * range_ratio
    
    return factor
