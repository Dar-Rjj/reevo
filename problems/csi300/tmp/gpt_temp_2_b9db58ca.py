import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate price slope (5-day linear regression on close prices)
    price_slope = df['close'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True
    )
    
    # Calculate volume slope (5-day linear regression on volume)
    volume_slope = df['volume'].rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True
    )
    
    # Compute divergence between price and volume trends
    divergence = (price_slope - volume_slope) * price_slope
    
    # Normalize by recent price volatility (10-day std dev)
    price_std = df['close'].rolling(window=10).std()
    normalized_divergence = divergence / price_std
    
    # Cap extreme values at ±2 standard deviations (30-day rolling)
    cap_std = normalized_divergence.rolling(window=30).std()
    capped_factor = normalized_divergence.clip(
        lower=-2 * cap_std,
        upper=2 * cap_std
    )
    
    return capped_factor
