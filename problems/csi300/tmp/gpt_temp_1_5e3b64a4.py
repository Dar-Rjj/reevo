import pandas as pd
import numpy as np
def heuristics_v2(df):
    import numpy as np
    import pandas as pd
    
    # Compute 5-day Price Slope using linear regression
    def linear_regression_slope(x):
        if len(x) < 2:
            return np.nan
        x = np.arange(len(x))
        return np.polyfit(x, x, 1)[0]
    
    # Calculate Price Trend
    price_slope = df['close'].rolling(window=5).apply(linear_regression_slope, raw=True)
    
    # Calculate Volume Trend
    volume_slope = df['volume'].rolling(window=5).apply(linear_regression_slope, raw=True)
    
    # Detect Divergence Signal
    divergence_signal = price_slope * (-volume_slope)
    
    return divergence_signal
