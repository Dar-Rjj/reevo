import pandas as pd
import pandas as pd
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate Price Rate of Change
    price_roc = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Price Acceleration
    price_acceleration = (price_roc - price_roc.shift(1)) / price_roc.shift(1)
    
    # Calculate Volume Rate of Change
    volume_roc = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Calculate Volume Acceleration
    volume_acceleration = (volume_roc - volume_roc.shift(1)) / volume_roc.shift(1)
    
    # Combine Signals
    combined_signal = price_acceleration * volume_acceleration
    
    # Apply z-score normalization
    normalized_signal = combined_signal.groupby(level=0).apply(zscore)
    
    return normalized_signal
