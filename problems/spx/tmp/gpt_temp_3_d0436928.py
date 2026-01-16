import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute intraday price change
    intraday_change = df['close'] - df['open']
    
    # Compute intraday range (high - low)
    intraday_range = df['high'] - df['low']
    
    # Avoid division by zero by replacing 0 ranges with NaN (will propagate)
    intraday_range.replace(0, np.nan, inplace=True)
    
    # Compute intraday momentum efficiency
    momentum_efficiency = intraday_change / intraday_range
    
    # Compute volume momentum (3-day rolling slope)
    volume = df['volume']
    rolling_slope = volume.rolling(3).apply(lambda x: (x[-1] - x[0]) / 2 if len(x) == 3 else np.nan)
    
    # Compute 10-day volume moving average
    volume_ma_10 = volume.rolling(10).mean()
    
    # Normalize volume slope by 10-day MA
    volume_momentum = rolling_slope / volume_ma_10
    
    # Adjust momentum efficiency by volume momentum
    factor = momentum_efficiency * volume_momentum
    
    # Scale factor between -1 and 1
    max_val = factor.abs().rolling(20, min_periods=1).max()  # Use 20-day lookback for scaling
    scaled_factor = factor / max_val.replace(0, 1)  # Avoid division by zero
    
    # Clip values to ensure they stay within [-1, 1]
    final_factor = scaled_factor.clip(-1, 1)
    
    return final_factor
