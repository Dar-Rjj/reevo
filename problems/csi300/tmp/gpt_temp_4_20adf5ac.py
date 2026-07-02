import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate normalized range
    current_range = (df['high'] - df['low']) / df['open']
    log_range = np.log1p(current_range)
    
    # Calculate historical 5-day range mean
    historical_ranges = []
    for i in range(len(df)):
        if i < 5:
            historical_ranges.append(np.nan)
            continue
        window = df.iloc[i-5:i]
        window_ranges = (window['high'] - window['low']) / window['open']
        historical_ranges.append(window_ranges.mean())
    
    historical_range_mean = pd.Series(historical_ranges, index=df.index)
    
    # Compute efficiency ratio
    efficiency_ratio = current_range / historical_range_mean - 1
    
    # Calculate volume surprise
    volume_medians = []
    for i in range(len(df)):
        if i < 10:
            volume_medians.append(np.nan)
            continue
        window_volumes = df['volume'].iloc[i-10:i]
        volume_medians.append(window_volumes.median())
    
    volume_median = pd.Series(volume_medians, index=df.index)
    volume_surprise = df['volume'] / volume_median
    
    # Combine components with sign from range direction
    range_direction = np.sign(df['close'] - df['open'])
    factor = efficiency_ratio * volume_surprise * range_direction
    
    return factor
