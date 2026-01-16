import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Range Ratio
    daily_range_ratio = (df['high'] - df['low']) / df['close']
    
    # Calculate weights for past 5 days (decay factor = 0.5)
    weights = np.exp(-0.5 * np.arange(5))
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Iterate through each day (starting from day 5 since we need 5-day lookback)
    for i in range(4, len(df)):
        current_date = df.index[i]
        
        # Get past 5 days data (including current day)
        past_5_days = df.iloc[i-4:i+1]
        
        # Calculate weighted volume sum for numerator
        weighted_volume_numerator = 0.0
        weighted_volume_denominator = 0.0
        
        for t in range(5):  # t=0 (current day) to t=4 (5 days ago)
            date_idx = i - (4 - t)  # maps t=0 to current day, t=4 to 5 days ago
            weight = weights[t]
            daily_vol = df.iloc[date_idx]['volume']
            daily_range = daily_range_ratio.iloc[date_idx]
            
            weighted_volume_numerator += daily_vol * daily_range * weight
            weighted_volume_denominator += daily_vol * weight
        
        # Calculate final factor value
        if weighted_volume_denominator != 0:
            factor_value = weighted_volume_numerator / weighted_volume_denominator
        else:
            factor_value = 0
        
        factor_values.at[current_date] = factor_value
    
    return factor_values
