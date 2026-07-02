import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(data):
    # Price Range Component
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['range_sma_5'] = data['price_range'].rolling(window=5, min_periods=1).mean()
    
    # Volume Confirmation
    data['avg_volume_20'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['avg_volume_20']
    
    # Calculate volume z-score using only historical data
    data['volume_z'] = data.groupby(data.index.date)['volume'].transform(
        lambda x: zscore(x, ddof=1) if len(x) > 1 else 0
    )
    
    # Combine components
    data['combined'] = data['range_sma_5'] * data['volume_z']
    
    # Generate signal (reversal factor)
    data['factor'] = -1 * data['combined']
    
    # Normalization (daily cross-sectional rank)
    data['factor_rank'] = data.groupby(data.index.date)['factor'].rank(pct=True)
    
    return data['factor_rank']
