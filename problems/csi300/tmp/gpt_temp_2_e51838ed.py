import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price range
    df['price_range'] = df['high'] - df['low']
    
    # Normalize intraday range
    df['normalized_range'] = df['price_range'] / df['close']
    
    # Compute 5-day rolling percentile of volume
    df['volume_percentile'] = df['volume'].rolling(window=5).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    # Multiply normalized range by volume percentile
    df['volatility_factor'] = df['normalized_range'] * df['volume_percentile']
    
    # Calculate breakout strength
    df['breakout_strength'] = (df['close'] - df['open']) / df['price_range']
    
    # Combine with volatility factor
    df['breakout_signal'] = df['breakout_strength'] * df['volatility_factor']
    
    # Scale by log(volume)
    df['heuristic_factor'] = df['breakout_signal'] * np.log(df['volume'])
    
    # Return the heuristic factor as a Series
    return df['heuristic_factor']
