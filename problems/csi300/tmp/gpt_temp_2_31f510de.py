import pandas as pd
import pandas as pd
from scipy.stats import skew

def heuristics_v2(df):
    # Calculate Middle Price
    df['middle_price'] = (df['high'] + df['low']) / 2
    
    # Calculate Close Price Relative Position
    df['relative_position'] = (df['close'] - df['middle_price']) / df['middle_price']
    
    # Calculate 10-day Rolling Volume Skewness
    df['volume_skewness'] = df['volume'].rolling(window=10).apply(skew)
    
    # Combine Signals
    df['momentum_adjusted'] = df['relative_position'] * df['volume_skewness']
    
    # Normalize by Liquidity
    df['abs_skewness'] = df['volume_skewness'].abs()
    df['factor'] = df['momentum_adjusted'] / df['abs_skewness']
    
    return df['factor']
