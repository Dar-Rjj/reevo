import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(data):
    # Calculate intraday return
    intraday_return = (data['close'] / data['open']) - 1
    
    # Calculate normalized price range
    price_range = (data['high'] - data['low']) / data['open']
    
    # Calculate price efficiency component
    price_efficiency = intraday_return / (price_range + 1e-6)  # Add small constant to avoid division by zero
    
    # Calculate volume z-score with 20-day rolling window
    volume_zscore = data['volume'].rolling(window=20).apply(lambda x: zscore(x, ddof=1)[-1], raw=True)
    
    # Combine components
    alpha = price_efficiency * volume_zscore
    
    # Rank normalization
    ranked_alpha = alpha.rank(pct=True)
    
    return ranked_alpha
