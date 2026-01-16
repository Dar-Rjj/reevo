import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Price Range
    price_range = df['high'] - df['low']
    
    # Price Efficiency
    price_efficiency = intraday_return / price_range.replace(0, np.nan)
    
    # Volume Z-Score
    volume_ma = df['volume'].rolling(window=10).mean()
    volume_std = df['volume'].rolling(window=10).std()
    volume_zscore = (df['volume'] - volume_ma) / volume_std.replace(0, np.nan)
    
    # Volume Trend
    def calculate_slope(series):
        if len(series) < 5:
            return np.nan
        return linregress(range(5), series[-5:])[0]
    
    volume_slope = df['volume'].rolling(window=5).apply(calculate_slope, raw=True)
    volume_divergence = volume_zscore * volume_slope
    
    # Combine Signals
    combined_signal = price_efficiency * volume_divergence
    
    # Scale by Volatility
    return_stddev = df['close'].pct_change().rolling(window=5).std()
    final_factor = combined_signal / return_stddev.replace(0, np.nan)
    
    return final_factor
