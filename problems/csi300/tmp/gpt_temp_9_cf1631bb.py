import pandas as pd
import pandas as pd
from scipy.stats import skew

def heuristics_v2(df):
    # Calculate High-to-Low Range
    df['range'] = df['high'] - df['low']
    
    # Normalize Range by Close Price
    df['normalized_range'] = df['range'] / df['close']
    
    # Calculate Skewness of Price Range
    rolling_window = 5
    df['skewness'] = df['range'].rolling(window=rolling_window).apply(skew)
    
    # Normalize Skewness
    mean_skewness = df['skewness'].rolling(window=rolling_window).mean()
    std_skewness = df['skewness'].rolling(window=rolling_window).std()
    df['normalized_skewness'] = (df['skewness'] - mean_skewness) / std_skewness
    
    # Multiply Intraday Momentum by Skewness Filter
    df['factor'] = df['normalized_range'] * df['normalized_skewness']
    
    return df['factor']
