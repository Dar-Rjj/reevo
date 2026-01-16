import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute Momentum
    rolling_mean = df['close'].rolling(window=5, min_periods=1).mean()
    momentum = df['close'] / rolling_mean
    
    # Compute Current Slope
    def get_current_slope(window):
        return linregress(np.arange(len(window)), window).slope
    current_slope = df['close'].rolling(window=3, min_periods=1).apply(get_current_slope)
    
    # Compute Momentum Slope
    momentum_slope = momentum.rolling(window=3, min_periods=1).apply(get_current_slope)
    
    # Compare Slopes
    divergence = current_slope - momentum_slope
    
    return divergence
