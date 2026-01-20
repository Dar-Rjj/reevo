import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Calculate High-to-Close Ratio
    high_to_close = (data['high'] - data['close']) / data['close']
    
    # Calculate Low-to-Close Ratio
    low_to_close = (data['low'] - data['close']) / data['close']
    
    # Calculate Rolling Volume Percentile
    rolling_volume = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Combine Momentum and Volume
    factor_high = high_to_close * rolling_volume
    factor_low = low_to_close * rolling_volume
    
    # Final factor is the average of high and low factors
    factor = (factor_high + factor_low) / 2
    
    return factor
