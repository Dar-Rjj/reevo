import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    def calculate_slope(x, window):
        slopes = np.zeros(len(x))
        for i in range(window-1, len(x)):
            y = x[i+1-window:i+1]
            slope, _, _, _, _ = linregress(range(window), y)
            slopes[i] = slope
        return slopes
    
    # Price Trend Component
    df['price_slope_5'] = calculate_slope(df['close'].values, 5)
    df['price_slope_20'] = calculate_slope(df['close'].values, 20)
    
    # Volume Trend Component
    df['volume_slope_5'] = calculate_slope(df['volume'].values, 5)
    df['volume_slope_20'] = calculate_slope(df['volume'].values, 20)
    
    # Divergence Signal
    df['divergence_signal'] = 0
    df.loc[(df['price_slope_5'] > 0) & (df['volume_slope_5'] < 0), 'divergence_signal'] = -1
    df.loc[(df['price_slope_5'] < 0) & (df['volume_slope_5'] > 0), 'divergence_signal'] = 1
    
    # Magnitude Adjustment
    df['trend_strength'] = abs(df['price_slope_20']) * abs(df['volume_slope_20'])
    df['price_volatility'] = df['close'].rolling(window=20).std()
    df['adjusted_signal'] = df['divergence_signal'] * df['trend_strength']
    df['normalized_signal'] = df['adjusted_signal'] / df['price_volatility']
    
    # Sigmoid Scaling
    df['factor'] = 1 / (1 + np.exp(-df['normalized_signal']))
    
    return df['factor']
