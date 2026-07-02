import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day Price Slope
    def calculate_price_slope(close_prices):
        return linregress(np.arange(5), close_prices).slope
    
    # Calculate 5-day Volume Slope
    def calculate_volume_slope(volumes):
        return linregress(np.arange(5), volumes).slope
    
    # Initialize the output series
    divergence_signal = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        # Extract the last 5 days of close prices and volumes
        close_prices = df['close'].iloc[i-4:i+1]
        volumes = df['volume'].iloc[i-4:i+1]
        
        # Calculate slopes
        price_slope = calculate_price_slope(close_prices)
        volume_slope = calculate_volume_slope(volumes)
        
        # Compute divergence signal
        divergence = price_slope * volume_slope
        
        # Apply sign correction
        if np.sign(price_slope) != np.sign(volume_slope):
            divergence *= -1
        
        divergence_signal.iloc[i] = divergence
    
    return divergence_signal
