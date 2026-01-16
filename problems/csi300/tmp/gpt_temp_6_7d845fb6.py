import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Detect Momentum Direction
    df['momentum'] = df['close'].diff(5)
    
    # Confirm with Volume
    df['volume_momentum'] = df['volume'].diff(5)
    
    # Detect Extreme Momentum using Z-Score
    window = 20
    df['momentum_zscore'] = df['momentum'].rolling(window=window).apply(lambda x: zscore(x)[-1], raw=True)
    
    # Intraday Reversal Signal
    df['mid_price'] = (df['high'] + df['low']) / 2
    factor = pd.Series(0, index=df.index)
    
    factor[((df['close'] < df['mid_price']) & (df['momentum_zscore'] > 2.0))] = 1
    factor[((df['close'] > df['mid_price']) & (df['momentum_zscore'] < -2.0))] = -1
    
    return factor
