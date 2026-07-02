import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price trends
    close = df['close']
    short_term_price_trend = close.rolling(5).apply(lambda x: np.polyfit(np.arange(5), x, 1)[0], raw=True)
    long_term_price_trend = close.rolling(20).apply(lambda x: np.polyfit(np.arange(20), x, 1)[0], raw=True)
    
    # Calculate volume trends
    volume = df['volume']
    short_term_volume_trend = volume.rolling(5).apply(lambda x: np.polyfit(np.arange(5), x, 1)[0], raw=True)
    long_term_volume_trend = volume.rolling(20).apply(lambda x: np.polyfit(np.arange(20), x, 1)[0], raw=True)
    
    # Calculate divergence
    divergence = (short_term_price_trend * long_term_volume_trend) - (long_term_price_trend * short_term_volume_trend)
    
    # Normalize
    close_std = close.rolling(20).std()
    normalized_divergence = (divergence / close_std) * 100
    
    return normalized_divergence
