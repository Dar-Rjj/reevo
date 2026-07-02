import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate 5-day Price Slope using linear regression
    price_slope = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        x = np.arange(5)
        y = df['close'].iloc[i-4:i+1].values
        slope, _, _, _, _ = linregress(x, y)
        price_slope.iloc[i] = slope
    
    # Calculate 5-day Volume Slope using linear regression
    volume_slope = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        x = np.arange(5)
        y = df['volume'].iloc[i-4:i+1].values
        slope, _, _, _, _ = linregress(x, y)
        volume_slope.iloc[i] = slope
    
    # Normalize Price Slope by 5-day Close price StdDev
    close_std = df['close'].rolling(window=5, min_periods=1).std()
    normalized_price_slope = price_slope / close_std
    
    # Normalize Volume Slope by 5-day Volume StdDev
    volume_std = df['volume'].rolling(window=5, min_periods=1).std()
    normalized_volume_slope = volume_slope / volume_std
    
    # Compute Divergence Signal
    divergence_signal = normalized_price_slope * np.sign(normalized_volume_slope)
    
    # Apply 3-day moving average smoothing
    factor = divergence_signal.rolling(window=3, min_periods=1).mean()
    
    return factor
