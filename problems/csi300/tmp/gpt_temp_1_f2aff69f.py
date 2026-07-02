import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate intraday range normalized by open price
    intraday_range = (df['high'] - df['low']) / df['open']
    
    # Calculate 10-day historical average of intraday range (excluding current day)
    historical_range = intraday_range.rolling(window=10, min_periods=1).mean()
    
    # Calculate compression ratio (current range vs historical)
    compression_ratio = intraday_range / historical_range
    
    # Calculate volume surprise (current volume vs 10-day MA)
    volume_ma = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_surprise = df['volume'] / volume_ma
    
    # Combine signals with volume weighting
    combined_signal = compression_ratio * volume_surprise
    
    # Apply z-score normalization using only historical data
    factor_values = combined_signal.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x[:-1].mean()) / x[:-1].std() if len(x) > 1 else np.nan
    )
    
    return factor_values
