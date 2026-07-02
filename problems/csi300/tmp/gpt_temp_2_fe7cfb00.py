import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate current day range
    df['current_range'] = df['high'] - df['low']
    
    # Calculate rolling 5-day average historical range
    df['historical_range'] = df['high'].rolling(window=5).apply(lambda x: (x - df['low'].shift(1).rolling(window=5).mean()).mean(), raw=False)
    
    # Compute range expansion ratio
    df['range_expansion'] = (df['current_range'] / df['historical_range']) - 1
    
    # Calculate volume spike
    df['volume_spike'] = df['volume'] / df['volume'].rolling(window=10).mean()
    
    # Combine components with close-to-open direction
    df['direction'] = (df['close'] - df['open']) / df['open']
    df['factor'] = df['range_expansion'] * df['volume_spike'] * df['direction'].apply(lambda x: 1 if x >= 0 else -1)
    
    # Return the factor series
    return df['factor']
