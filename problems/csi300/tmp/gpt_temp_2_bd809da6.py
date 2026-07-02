import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate price trend (5-day linear regression slope)
    price_slopes = []
    for i in range(len(df)):
        if i < 5:
            price_slopes.append(np.nan)
            continue
        window = df['close'].iloc[i-5:i]
        slope = linregress(np.arange(5), window).slope
        price_slopes.append(slope)
    df['price_slope'] = price_slopes
    
    # Calculate volume trend (5-day linear regression slope)
    volume_slopes = []
    for i in range(len(df)):
        if i < 5:
            volume_slopes.append(np.nan)
            continue
        window = df['volume'].iloc[i-5:i]
        slope = linregress(np.arange(5), window).slope
        volume_slopes.append(slope)
    df['volume_slope'] = volume_slopes
    
    # Normalize price slope
    df['norm_price_slope'] = (df['price_slope'] / df['close'].shift(5)) * 100
    
    # Normalize volume slope
    df['norm_volume_slope'] = (df['volume_slope'] / df['volume'].shift(5)) * 100
    
    # Generate signals
    signals = pd.Series(0, index=df.index)
    
    # Bullish divergence condition
    bullish_mask = (df['norm_price_slope'] > 0.5) & (df['norm_volume_slope'] < -0.3)
    signals[bullish_mask] = 1
    
    # Bearish divergence condition
    bearish_mask = (df['norm_price_slope'] < -0.5) & (df['norm_volume_slope'] > 0.3)
    signals[bearish_mask] = -1
    
    return signals
