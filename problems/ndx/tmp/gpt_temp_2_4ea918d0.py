import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress, skew

def heuristics_v2(data):
    # Initialize output Series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Parameters
    trend_window = 20  # for trend momentum calculation
    skewness_window = 10  # for return skewness calculation
    
    # Calculate daily returns
    daily_returns = data['close'].pct_change()
    
    for t in range(1, len(data)):
        current_date = data.index[t]
        
        # ===== Trend Momentum Component =====
        if t >= trend_window:
            # Get past close prices (current and historical)
            close_prices = data['close'].iloc[t-trend_window+1:t+1].values
            
            # Calculate linear regression slope
            x = np.arange(len(close_prices))
            slope, _, _, _, _ = linregress(x, close_prices)
            trend_momentum = slope
        else:
            trend_momentum = 0
        
        # ===== Return Skewness Component =====
        if t >= skewness_window:
            # Get past returns (current and historical)
            returns = daily_returns.iloc[t-skewness_window+1:t+1].values
            
            # Calculate skewness of returns
            return_skewness = skew(returns)
        else:
            return_skewness = 0
        
        # ===== Combine Components =====
        # Simple weighted combination (can be adjusted)
        factor_value = 0.6 * trend_momentum + 0.4 * return_skewness
        factor_values.at[current_date] = factor_value
    
    return factor_values
