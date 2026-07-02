import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Intraday Price Direction
    intraday_price_direction = (df['close'] - df['open']) / df['open']
    
    # Price Direction Persistence
    rolling_zscore = intraday_price_direction.rolling(window=7).apply(lambda x: zscore(x)[-1], raw=True)
    price_direction_persistence = rolling_zscore * np.sign(intraday_price_direction)
    
    # Volume Deviation
    volume_deviation = df['volume'] - df['volume'].rolling(window=20).mean()
    
    # Volume-Price Interaction
    volume_price_interaction = volume_deviation * price_direction_persistence
    
    # Rank Normalization
    factor = volume_price_interaction.rank(pct=True)
    
    return factor
