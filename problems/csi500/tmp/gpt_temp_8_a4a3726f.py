import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Midpoint Price
    midpoint = (df['high'] + df['low']) / 2
    
    # Compute Price Deviation from Midpoint
    deviation = df['close'] - midpoint
    
    # Calculate Volume-Adjusted Deviation
    volume_avg = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_adjusted_deviation = deviation * df['volume'] / volume_avg
    
    # Incorporate Opening Gap
    open_ratio = df['open'] / df['close'].shift(1)
    gap_adjusted_factor = volume_adjusted_deviation * open_ratio
    
    # Apply Momentum Filter
    def calc_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return slope
    
    # Calculate 5-day Price Trend
    trend = df['close'].rolling(window=5, min_periods=2).apply(calc_slope, raw=False)
    
    # Multiply by Combined Factor
    factor = gap_adjusted_factor * (1 / (1 + np.abs(trend)))
    
    return factor
