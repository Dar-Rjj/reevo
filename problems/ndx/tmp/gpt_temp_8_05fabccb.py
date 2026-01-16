import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Daily Normalized Range
    df['daily_normalized_range'] = (df['high'] - df['low']) / df['close']
    
    # Volume-Weighted Signal
    rolling_period = 5
    df['volume_weighted_signal'] = (
        df['volume'].rolling(window=rolling_period).apply(lambda x: (x * df.loc[x.index, 'daily_normalized_range']).sum()) /
        df['volume'].rolling(window=rolling_period).sum()
    )
    
    # Rolling StdDev of Close
    df['rolling_stddev'] = df['close'].rolling(window=rolling_period).std()
    
    # Volume-Adjusted Range Momentum
    factor = df['volume_weighted_signal'] / df['rolling_stddev']
    
    return factor
