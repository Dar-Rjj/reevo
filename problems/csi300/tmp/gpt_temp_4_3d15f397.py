import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Ensure we have required columns
    if not all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
        raise ValueError("DataFrame must contain 'close', 'high', 'low', and 'volume' columns")
    
    # Initialize result Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Price Trend Component: 5-day Price Slope
    price_slope = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        window = df['close'].iloc[i-4:i+1]
        if len(window) < 2:
            continue
        slope = linregress(np.arange(len(window)), window.values).slope
        price_slope.iloc[i] = slope
    
    # Volume Intensity Component
    volume_amplitude = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        # Calculate High-Low Range
        high_low_range = df['high'].iloc[i] - df['low'].iloc[i]
        
        # Calculate 5-day Average Volume (including current day)
        avg_volume = df['volume'].iloc[i-4:i+1].mean()
        
        # Avoid division by zero
        if avg_volume == 0:
            volume_amplitude.iloc[i] = 0
        else:
            volume_amplitude.iloc[i] = high_low_range / avg_volume
    
    # Formulate Divergence Signal
    for i in range(4, len(df)):
        if pd.isna(price_slope.iloc[i]) or pd.isna(volume_amplitude.iloc[i]):
            continue
        
        divergence = price_slope.iloc[i] - volume_amplitude.iloc[i]
        
        # Normalize by Price Level (divide by close price)
        if df['close'].iloc[i] == 0:
            factor.iloc[i] = 0
        else:
            factor.iloc[i] = divergence / df['close'].iloc[i]
    
    return factor
