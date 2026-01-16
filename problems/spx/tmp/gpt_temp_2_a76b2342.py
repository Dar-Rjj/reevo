import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute Intraday Momentum: (High - Low) / Close
    intraday_momentum = (df['high'] - df['low']) / df['close']
    
    # Calculate Volume Trend: 5-day linear regression slope of Volume
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: linregress(np.arange(len(x)), x)[0] if len(x) == 5 else np.nan
    )
    
    # Adjust Momentum by Volume Activity
    volume_adjusted_momentum = intraday_momentum * volume_trend
    
    # Normalize by Volatility: 5-day rolling std of Close
    volatility = df['close'].rolling(window=5).std()
    
    # Final factor: Volume-Adjusted Momentum / Volatility
    factor = volume_adjusted_momentum / volatility
    
    return factor
