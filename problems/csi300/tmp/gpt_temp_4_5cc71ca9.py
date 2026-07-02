import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import linregress

def heuristics_v2(df):
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Prepare rolling windows
    window_size = 5
    std_window = 20
    
    for i in range(std_window, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Calculate price slope (5-day regression)
        if i >= window_size:
            price_window = current_data['close'].iloc[-window_size:]
            x = np.arange(len(price_window))
            price_slope = linregress(x, price_window.values)[0]
            
            # Calculate volume slope (5-day regression)
            volume_window = current_data['volume'].iloc[-window_size:]
            volume_slope = linregress(x, volume_window.values)[0]
            
            # Calculate rolling standard deviations
            price_std = current_data['close'].iloc[-std_window:].std()
            volume_std = current_data['volume'].iloc[-std_window:].std()
            
            # Normalize slopes
            if price_std > 0:
                norm_price_slope = price_slope / price_std
            else:
                norm_price_slope = 0
                
            if volume_std > 0:
                norm_volume_slope = volume_slope / volume_std
            else:
                norm_volume_slope = 0
            
            # Calculate divergence factor
            divergence = norm_price_slope * (-norm_volume_slope)
            
            # Apply sigmoid scaling
            factor.iloc[i] = 1 / (1 + np.exp(-divergence))
        else:
            factor.iloc[i] = 0
    
    return factor
