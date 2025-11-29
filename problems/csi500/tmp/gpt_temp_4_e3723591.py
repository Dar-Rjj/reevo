import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price components
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling windows (using only past data)
    for i in range(len(df)):
        if i < 10:  # Need at least 10 days of history
            factor.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only current and past data
        
        # 1. Calculate Breakout Ratio: (Close - Low) / (High - Low)
        breakout_ratio = (close.iloc[i] - low.iloc[i]) / (high.iloc[i] - low.iloc[i] + 1e-8)
        
        # 2. Volatility Expansion Adjustment
        # Current Range vs 10-day Average Range
        current_range = high.iloc[i] - low.iloc[i]
        avg_range = np.mean([high.iloc[j] - low.iloc[j] for j in range(i-9, i+1)])
        expansion_ratio = current_range / (avg_range + 1e-8)
        
        # Volatility-Adjusted Breakout
        vol_adjusted_breakout = breakout_ratio * expansion_ratio
        
        # 3. Volume Confirmation
        # Calculate Volume Ratio: Volume / 10-day Volume Mean
        current_volume = volume.iloc[i]
        avg_volume = np.mean([volume.iloc[j] for j in range(i-9, i+1)])
        volume_ratio = current_volume / (avg_volume + 1e-8)
        
        # Apply Volume Adjustment
        if volume_ratio > 1.2:  # High Volume
            volume_adjusted_breakout = vol_adjusted_breakout
        else:  # Normal Volume
            volume_adjusted_breakout = vol_adjusted_breakout * volume_ratio
        
        # 4. Range Expansion Momentum
        range_expansion = (high.iloc[i] - low.iloc[i]) / (close.iloc[i-1] + 1e-8)
        
        # 5. Concentration Confirmation
        concentration = amount.iloc[i] / (volume.iloc[i] + 1e-8)
        
        # Generate Composite Factor
        composite_factor = volume_adjusted_breakout * range_expansion * concentration
        
        factor.iloc[i] = composite_factor
    
    return factor
