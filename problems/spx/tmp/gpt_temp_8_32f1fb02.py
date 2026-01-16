import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Initialize output Series
    factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 4:  # Need at least 5 days of data
            factor.iloc[i] = np.nan
            continue
        
        current_data = df.iloc[i]
        past_data = df.iloc[i-4:i+1]  # t-4 to t (5 days)
        
        # Price Momentum Component
        momentum = (current_data['close'] / df.iloc[i-4]['close']) - 1
        
        # Adjust by Intraday Range
        daily_range = (current_data['high'] - current_data['low']) / current_data['close']
        price_component = momentum * daily_range
        
        # Volume Confirmation Component
        # Volume Stability Signal
        vol_std = past_data['volume'].std()
        vol_mean = past_data['volume'].mean()
        vol_stability = vol_std / vol_mean if vol_mean != 0 else 0
        
        # Volume Ratio Signal
        vol_ratio = current_data['volume'] / vol_mean if vol_mean != 0 else 0
        
        # Combine components
        volume_component = vol_stability * vol_ratio
        
        # Final factor value
        factor.iloc[i] = price_component * volume_component
    
    return factor
