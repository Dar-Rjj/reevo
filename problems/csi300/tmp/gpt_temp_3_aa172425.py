import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Initialize output Series
    factor = pd.Series(index=df.index, dtype=np.float64)
    
    for i in range(1, len(df)):
        current = df.iloc[i]
        past = df.iloc[:i]  # Only historical data up to current day
        
        # Mean Reversion Component
        midpoint = (current['high'] + current['low']) / 2
        deviation = current['close'] - midpoint
        daily_range = current['high'] - current['low']
        if daily_range != 0:
            normalized_deviation = deviation / daily_range
        else:
            normalized_deviation = 0
        
        # Volume Clustering Component
        vol_ma = past['volume'].rolling(10).mean().iloc[-1] if i >= 10 else np.nan
        if not np.isnan(vol_ma) and vol_ma != 0:
            volume_ratio = current['volume'] / vol_ma
            volume_component = np.log(volume_ratio)
        else:
            volume_component = 0
        
        # Combine signals (avoid future lookahead)
        combined_signal = normalized_deviation * volume_component
        
        # Assign factor value
        factor.iloc[i] = combined_signal
    
    # Fill first value (can't calculate without history)
    if len(factor) > 0:
        factor.iloc[0] = 0
    
    return factor
