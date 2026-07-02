import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Initialize output Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Pre-calculate required periods
    min_periods = max(5, 3)  # We need at least 3 for slopes and 5 for correlation
    
    for i in range(min_periods - 1, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Price Trend Component
        if i >= 2:  # Need at least 3 days for slope
            price_window = current_data['close'].iloc[-3:]
            price_slope = linregress(range(3), price_window).slope
            price_component = price_slope / current_data['close'].iloc[-1]
        else:
            price_component = 0
        
        # Volume Trend Component
        if i >= 2:  # Need at least 3 days for slope
            volume_window = current_data['volume'].iloc[-3:]
            volume_slope = linregress(range(3), volume_window).slope
            volume_component = volume_slope / current_data['volume'].iloc[-1]
        else:
            volume_component = 0
        
        # Divergence Signal
        if i >= 4:  # Need at least 5 days for correlation
            # Calculate rolling price and volume slopes for correlation
            price_slopes = []
            volume_slopes = []
            
            for j in range(i-4, i+1):
                if j >= 2:  # Need at least 3 points for slope
                    price_slope = linregress(range(3), current_data['close'].iloc[j-2:j+1]).slope
                    volume_slope = linregress(range(3), current_data['volume'].iloc[j-2:j+1]).slope
                else:
                    price_slope = 0
                    volume_slope = 0
                
                price_slopes.append(price_slope)
                volume_slopes.append(volume_slope)
            
            correlation = np.corrcoef(price_slopes, volume_slopes)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            divergence = price_component * (1 - correlation)
        else:
            divergence = 0
        
        factor.iloc[i] = divergence
    
    return factor
