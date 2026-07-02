import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Divergence Component
    intraday_range = (df['high'] - df['low']) / df['open'] * 100
    rolling_median_range = intraday_range.rolling(window=5, min_periods=1).median()
    price_divergence = (intraday_range / rolling_median_range) - 1

    # Volume Divergence Component
    rolling_median_volume = df['volume'].rolling(window=5, min_periods=1).median()
    volume_spike = np.log(df['volume'] / rolling_median_volume)
    
    # Calculate 5-day volume slope
    def calculate_slope(x):
        if len(x) < 2:
            return np.nan
        return linregress(np.arange(len(x)), x).slope
    
    volume_slope = df['volume'].rolling(window=5, min_periods=1).apply(calculate_slope, raw=True)
    volume_divergence = volume_spike * volume_slope

    # Combined Factor
    combined_factor = price_divergence * volume_divergence
    sigmoid_factor = 1 / (1 + np.exp(-combined_factor))
    
    return sigmoid_factor
