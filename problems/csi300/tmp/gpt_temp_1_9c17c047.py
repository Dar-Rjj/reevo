import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Price Trend
    price_slope = df['close'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x)[0])
    normalized_price_slope = price_slope / df['close']
    
    # Calculate Volume Trend
    volume_slope = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(5), x)[0])
    normalized_volume_slope = volume_slope / df['volume']
    
    # Detect Divergence
    correlation = normalized_price_slope.rolling(window=5).corr(normalized_volume_slope)
    
    # Generate Signal
    signal = np.where(correlation < -0.5, 1, 0)
    
    return pd.Series(signal, index=df.index)
