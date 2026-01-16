import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate High-Low Range
    df['range'] = df['high'] - df['low']
    
    # Compute Rolling Momentum
    momentum = []
    for i in range(len(df)):
        if i < 2:
            momentum.append(np.nan)
            continue
        window = df['range'].iloc[max(i-2, 0):i+1].values
        slope = linregress(np.arange(len(window)), window).slope
        momentum.append(slope)
    df['momentum'] = momentum
    
    # Normalize Momentum
    rolling_std = df['range'].rolling(window=3, min_periods=1).std()
    df['normalized_momentum'] = df['momentum'] / (df['range'] * rolling_std)
    
    return df['normalized_momentum']
