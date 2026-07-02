import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate intraday price range
    df['price_range'] = df['high'] - df['low']
    
    # Normalize intraday range by close price
    df['normalized_range'] = df['price_range'] / df['close']
    
    # Compute 5-day rolling percentile rank of volume
    df['volume_percentile'] = df['volume'].rolling(window=5).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    # Combine normalized range with volume percentile
    df['scaled_range'] = df['normalized_range'] * df['volume_percentile']
    
    # Calculate midpoint return
    df['midpoint_return'] = (df['close'] - (df['high'] + df['low']) / 2) / df['price_range']
    
    # Combine midpoint return with scaled range and log(volume)
    df['factor'] = df['midpoint_return'] * df['scaled_range'] * np.log(df['volume'] + 1)
    
    # Return the factor series indexed by date
    return df['factor']
