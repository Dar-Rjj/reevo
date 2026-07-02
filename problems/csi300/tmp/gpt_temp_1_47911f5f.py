import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Prepare rolling window for calculations
    window_size = 5
    
    # Calculate price slope (5-day linear regression slope of close prices)
    def calc_slope(series):
        x = np.arange(len(series))
        y = series.values
        cov = np.cov(x, y)
        return cov[0, 1] / cov[0, 0] if cov[0, 0] != 0 else 0
    
    price_slope = df['close'].rolling(window=window_size, min_periods=window_size).apply(calc_slope, raw=False)
    
    # Calculate volume slope (5-day linear regression slope of volume)
    volume_slope = df['volume'].rolling(window=window_size, min_periods=window_size).apply(calc_slope, raw=False)
    
    # Calculate divergence factor
    divergence = price_slope * volume_slope
    factor = np.sign(divergence)
    
    return factor
