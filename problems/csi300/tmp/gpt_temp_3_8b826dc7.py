import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Current Day's Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Compute 5-day Rolling Minimum Price Range
    df['min_price_range'] = df['price_range'].rolling(window=5, min_periods=1).min()
    
    # Measure Price Compression
    df['price_compression'] = df['price_range'] / df['min_price_range']
    
    # Calculate Volume Divergence
    df['rolling_median_volume'] = df['volume'].rolling(window=5, min_periods=1).median()
    df['volume_divergence'] = df['volume'] - df['rolling_median_volume']
    
    # Calculate 5-day MAD of Volume
    df['volume_mad'] = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: np.median(np.abs(x - np.median(x))))
    
    # Normalize Volume Divergence by Historical Volume Volatility
    df['normalized_volume_divergence'] = df['volume_divergence'] / df['volume_mad']
    
    # Combine Signals
    df['combined_signal'] = df['price_compression'] * df['normalized_volume_divergence']
    
    # Calculate 5-day High-Low Price Range StdDev
    df['price_range_stddev'] = df['price_range'].rolling(window=5, min_periods=1).std()
    
    # Normalize Combined Signal by Recent Price Volatility
    df['factor'] = df['combined_signal'] / df['price_range_stddev']
    
    return df['factor']
