import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = (df['close'] - df['open']) / df['open']
    
    # Normalize by Volatility (5-day rolling std dev of close)
    rolling_std = df['close'].rolling(window=5).std()
    normalized_return = intraday_return / rolling_std
    
    # Calculate Volume Slope (Linear regression slope of 5-day volume)
    volume_slope = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x).slope, raw=False)
    
    # Cap extreme values of volume slope
    volume_slope = volume_slope.clip(lower=0.0, upper=2.0)
    
    # Combine Signals
    combined_signal = normalized_return * volume_slope
    
    # Apply Directional Filter
    directional_filter = np.where(df['close'] > df['open'], 1, -1)
    final_signal = combined_signal * directional_filter
    
    return final_signal
