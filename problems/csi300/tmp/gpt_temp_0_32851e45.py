import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    # Initialize result Series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate price trends
    data['price_5d_slope'] = data['close'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True)
    data['price_20d_slope'] = data['close'].rolling(window=20).apply(
        lambda x: linregress(np.arange(20), x)[0], raw=True)
    
    # Calculate volume trends
    data['volume_5d_slope'] = data['volume'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True)
    data['volume_20d_slope'] = data['volume'].rolling(window=20).apply(
        lambda x: linregress(np.arange(20), x)[0], raw=True)
    
    # Calculate volume standard deviation
    data['volume_20d_std'] = data['volume'].rolling(window=20).std()
    
    # Generate signals
    for t in range(20, len(data)):
        # Get current values
        p5 = data['price_5d_slope'].iloc[t]
        p20 = data['price_20d_slope'].iloc[t]
        v5 = data['volume_5d_slope'].iloc[t]
        v20 = data['volume_20d_slope'].iloc[t]
        v_std = data['volume_20d_std'].iloc[t]
        
        # Determine price trend direction (weighted combination)
        price_trend = 0.4 * p5 + 0.6 * p20
        
        # Determine volume trend direction (weighted combination)
        volume_trend = 0.4 * v5 + 0.6 * v20
        
        # Generate divergence signal
        if price_trend > 0 and volume_trend < 0:
            signal = -1
        elif price_trend < 0 and volume_trend > 0:
            signal = 1
        else:
            signal = 0
        
        # Adjust magnitude
        if v_std > 0:
            magnitude = signal * abs(price_trend) / v_std
        else:
            magnitude = 0
        
        factor.iloc[t] = magnitude
    
    return factor
