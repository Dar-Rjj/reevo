import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute 5-day Momentum
    momentum_5 = df['close'] - df['close'].shift(5)
    
    # Compute 10-day Momentum
    momentum_10 = df['close'] - df['close'].shift(10)
    
    # Calculate Acceleration
    acceleration = (momentum_5 - momentum_10) / df['close']
    
    # Compute Volume Trend Strength
    volume_trend_strength = df['volume'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x).slope / x.mean(), raw=True
    )
    
    # Adjust Acceleration with Volume Trend Strength
    factor = acceleration * volume_trend_strength
    
    return factor
