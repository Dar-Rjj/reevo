import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Compute Normalized Momentum
    close = data['close']
    momentum = close / close.shift(5) - 1  # 5-day return
    volatility = close.rolling(5).std()  # 5-day std dev
    normalized_momentum = momentum / volatility
    
    # Compute Volume-Adjusted Normalized Range
    daily_range = data['high'] - data['low']
    volume_adjusted_range = daily_range * data['volume'] / close
    
    # Combine components
    factor = normalized_momentum * volume_adjusted_range
    
    # Apply Z-Score normalization (cross-sectional)
    def zscore(series):
        return (series - series.mean()) / series.std()
    
    # Group by date and apply zscore (cross-sectional normalization)
    factor = factor.groupby(factor.index).transform(zscore)
    
    return factor
