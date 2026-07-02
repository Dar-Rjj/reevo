import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate intraday high-to-low range
    df['intraday_range'] = df['high'] - df['low']
    
    # Normalize intraday range by open price
    df['normalized_range'] = df['intraday_range'] / df['open']
    
    # Compute rolling 3-day percentile rank of volume
    df['volume_percentile'] = df['volume'].rolling(window=3).apply(lambda x: x.rank(pct=True).iloc[-1])
    
    # Weight normalized range by percentile rank of volume
    df['intraday_momentum'] = df['normalized_range'] * df['volume_percentile']
    
    # Calculate rolling 3-day return
    df['daily_return'] = df['close'].pct_change()
    df['rolling_return'] = df['daily_return'].rolling(window=3).sum()
    
    # Weighted reversal score
    df['reversal_score'] = df['intraday_momentum'] * df['rolling_return']
    
    # Use the sign of the reversal score to confirm direction
    df['factor'] = np.sign(df['reversal_score'])
    
    return df['factor']
