import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Measure Intraday Price Compression
    high_low_range = df['high'] - df['low']
    normalized_range = high_low_range / df['open']
    
    # Reversal Character
    rolling_median = df['close'].rolling(window=5, min_periods=1).median()
    rolling_mad = df['close'].rolling(window=5, min_periods=1).apply(lambda x: (x - x.median()).abs().mean())
    deviation = (df['close'] - rolling_median) / rolling_mad
    
    # Combine factors
    factor = -normalized_range * deviation
    
    return factor
