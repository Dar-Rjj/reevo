import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price changes
    df['price_change'] = (df['close'] - df['open']) / df['open']
    
    # Short-term price change (3-day rolling z-score)
    short_term = df['price_change'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Long-term price change (10-day rolling z-score)
    long_term = df['price_change'].rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Price divergence component (short - long)
    price_divergence = short_term - long_term
    
    # Volume divergence (5-day rolling z-score)
    volume_z = df['volume'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Combined signal (price divergence * volume divergence)
    factor = price_divergence * volume_z
    
    return factor
