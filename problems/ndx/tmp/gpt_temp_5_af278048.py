import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Short-Term Momentum: 5-Day Price Change
    short_term_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate Long-Term Trend Slope: 20-Day Rolling Slope
    def rolling_slope(series, window):
        slopes = np.zeros(len(series))
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series[i-window+1:i+1].values
                x = np.arange(len(y))
                slope = linregress(x, y)[0]
                slopes[i] = slope
        return pd.Series(slopes, index=series.index)
    
    long_term_slope = rolling_slope(df['close'], 20)
    
    # Calculate Log-Scaled Divergence
    divergence = short_term_momentum - long_term_slope
    log_divergence = np.log(np.abs(divergence))
    
    # Return the factor values
    return log_divergence
