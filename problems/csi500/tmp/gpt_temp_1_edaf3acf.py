import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    # Intraday Price Range Momentum
    df['price_range_ratio'] = (df['high'] - df['low']) / df['low']
    
    # 5-day price range slope
    df['price_range_5d_slope'] = df['price_range_ratio'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
    )
    
    # 20-day price range slope
    df['price_range_20d_slope'] = df['price_range_ratio'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
    )
    
    # Close Price Momentum
    # 5-day close price slope
    df['close_5d_slope'] = df['close'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
    )
    
    # 20-day close price slope
    df['close_20d_slope'] = df['close'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
    )
    
    # Volume Momentum Component
    # Volume Level Momentum
    # 5-day volume slope
    df['volume_5d_slope'] = df['volume'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
    )
    
    # 20-day volume slope
    df['volume_20d_slope'] = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
    )
    
    # Intraday Volume Pattern
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_5d_avg']
    df['volume_pattern'] = np.log(df['volume_ratio'].replace(0, 1e-6))
    
    # Divergence Combination
    # Short-term Divergence
    df['short_term_div'] = (df['price_range_5d_slope'] / df['volume_5d_slope'].replace(0, 1e-6)) * df['volume_pattern']
    
    # Medium-term Divergence
    df['medium_term_div'] = (df['price_range_20d_slope'] / df['volume_20d_slope'].replace(0, 1e-6)) * df['volume_pattern']
    
    # Signal Integration
    # Combine short-term and medium-term divergences
    df['combined_div'] = 0.6 * df['short_term_div'] + 0.4 * df['medium_term_div']
    
    # Apply 3-day smoothing
    factor = df['combined_div'].rolling(window=3).mean()
    
    return factor
