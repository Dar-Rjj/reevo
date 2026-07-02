import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Make a copy to avoid modifying original dataframe
    df = df.copy()
    
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1]  # All data up to current day
        
        # Price Trend Component
        price_window = current_data['close'].iloc[-5:]
        x = np.arange(len(price_window))
        price_slope = np.polyfit(x, price_window, 1)[0]
        price_component = price_slope / current_data['close'].iloc[-1]
        
        # Volume Trend Component
        volume_window = current_data['volume'].iloc[-5:]
        volume_slope = np.polyfit(x, volume_window, 1)[0]
        avg_volume = current_data['volume'].iloc[-5:].mean()
        volume_component = np.log1p(volume_slope / avg_volume)
        
        # Divergence Signal
        divergence = price_component * volume_component
        
        # Store the factor value for current day
        factor.iloc[i] = divergence
    
    # Cross-sectional rank and scale to [0,1]
    if len(factor) > 0:
        factor = factor.rank(pct=True)
    
    return factor
