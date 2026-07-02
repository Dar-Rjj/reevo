import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate intraday momentum (high-low range normalized by open)
    intraday_range = df['high'] - df['low']
    normalized_momentum = intraday_range / df['open']
    
    # Calculate volume change (today's volume relative to 5-day MA)
    volume_ma = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_change = df['volume'] / volume_ma
    
    # Combine signals
    combined_signal = normalized_momentum * volume_change
    
    # Apply 3-day z-score normalization (using only past and current data)
    factor = combined_signal.rolling(window=3, min_periods=1).apply(
        lambda x: zscore(x, ddof=1)[-1] if len(x) >= 2 else np.nan
    )
    
    return factor
