import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Measure Price Momentum
    df['Momentum'] = (df['close'] - df['open']) / df['open']
    df['Momentum_Std'] = df['Momentum'].rolling(window=10, min_periods=1).std()
    df['Normalized_Momentum'] = df['Momentum'] / df['Momentum_Std']
    
    # Detect Turnover Divergence
    df['MA20_Volume'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['Turnover_Ratio'] = df['volume'] / df['MA20_Volume']
    df['Divergence'] = df['Normalized_Momentum'] * np.sign(df['Turnover_Ratio'] - 1)
    
    # Adjust for Volume Trend
    def volume_trend(volume):
        if len(volume) < 2:
            return 0
        x = np.arange(len(volume))
        slope, _, _, _, _ = linregress(x, volume)
        return slope
    
    df['Volume_Trend'] = df['volume'].rolling(window=10, min_periods=1).apply(volume_trend, raw=False)
    df['Adjusted_Divergence'] = df['Divergence'] * df['Volume_Trend']
    
    # Generate Final Signal
    df['Signal_Mean'] = df['Adjusted_Divergence'].rolling(window=15, min_periods=1).mean()
    df['Signal_Std'] = df['Adjusted_Divergence'].rolling(window=15, min_periods=1).std()
    df['Final_Signal'] = (df['Adjusted_Divergence'] - df['Signal_Mean']) / df['Signal_Std']
    
    # Apply Thresholds
    df['Factor'] = np.where(df['Final_Signal'] > 2, 1, np.where(df['Final_Signal'] < -2, -1, 0))
    
    return df['Factor']
