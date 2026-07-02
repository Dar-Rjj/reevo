import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum (PM)
    price_change = df['close'].diff()
    rmar = price_change.rolling(window=10).mean()
    rmar_normalized = (rmar - rmar.mean()) / rmar.std()
    
    # Threshold for positive momentum
    threshold = 0.75 * rmar.median()
    pm = rmar_normalized.where(rmar >= threshold, 0)

    # Volume-Weighted Velocity (VWV)
    vapc = price_change * df['volume']
    vapc_zscore = vapc.rolling(window=20).apply(lambda x: (x[-1] - x.mean()) / x.std())
    
    volume_median = df['volume'].rolling(window=30).median()
    vsd = df['volume'] / volume_median
    
    vwv = vapc_zscore * vsd
    vwv_normalized = (vwv - vwv.mean()) / vwv.std()

    # Final Factor Construction
    final_factor = 0.7 * pm + 0.3 * vwv_normalized
    
    return final_factor
