import pandas as pd
import pandas as pd
from scipy.stats import skew

def heuristics_v2(df):
    # Intraday Momentum Calculation
    df['momentum'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Volume-Adjusted Momentum
    df['volume_ma'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['adjusted_momentum'] = df['momentum'] * df['volume_ratio']
    
    # Skewness Measurement and Adjustment
    df['rolling_skewness'] = df['adjusted_momentum'].rolling(window=5).apply(lambda x: skew(x))
    mean_skewness = df['rolling_skewness'].mean()
    std_skewness = df['rolling_skewness'].std()
    df['normalized_skewness'] = (df['rolling_skewness'] - mean_skewness) / std_skewness
    
    return df['normalized_skewness']
