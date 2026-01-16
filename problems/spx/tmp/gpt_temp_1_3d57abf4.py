import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate price efficiency
    actual_move = df['close'] - df['open']
    potential_move = df['high'] - df['low']
    efficiency_ratio = actual_move / potential_move.replace(0, np.nan)  # avoid division by zero
    
    # Calculate volume percentile (20-day rolling)
    volume_percentile = df['volume'].rolling(window=20).rank(pct=True)
    
    # Combine signals
    combined = efficiency_ratio * volume_percentile
    
    # Standardize the combined signal (20-day rolling z-score)
    factor = combined.rolling(window=20).apply(lambda x: zscore(x, ddof=1)[-1] if len(x) == 20 else np.nan)
    
    return factor
