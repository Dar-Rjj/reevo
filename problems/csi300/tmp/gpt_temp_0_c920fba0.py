import pandas as pd
import pandas as pd
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate Intraday Return
    intraday_return = abs(df['close'] / df['open'] - 1)
    
    # Normalize by Trading Range
    trading_range = (df['high'] - df['low']) / df['open']
    price_efficiency = intraday_return / trading_range
    
    # Calculate Volume Weight
    volume_ma = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_weight = df['volume'] / volume_ma
    
    # Combine Signals
    factor = price_efficiency * volume_weight
    
    # Apply z-score normalization
    factor_normalized = factor.groupby(level=0).apply(lambda x: zscore(x, ddof=1))
    
    return factor_normalized
