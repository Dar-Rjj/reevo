import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Component
    # Calculate 5-day linear regression slope on close prices
    close_prices = df['close']
    price_slopes = close_prices.rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True
    )
    # Normalize price trend
    normalized_price_trend = (price_slopes / close_prices) * 100
    
    # Volume Component
    # Calculate 5-day EMA slope on volume
    volume = df['volume']
    ema_volume = volume.ewm(span=5, adjust=False).mean()
    volume_slopes = ema_volume.diff()
    # Normalize volume trend
    normalized_volume_trend = (volume_slopes / volume) * 100
    
    # Divergence Signal
    # Initialize signal series
    signal = pd.Series(0, index=df.index)
    
    # Positive signal when price down and volume up
    signal[(price_slopes < 0) & (volume_slopes > 0)] = 1
    
    # Negative signal when price up and volume down
    signal[(price_slopes > 0) & (volume_slopes < 0)] = -1
    
    # Volume-weighted magnitude
    vol_std = volume.rolling(window=20).std()
    factor = signal * normalized_volume_trend.abs() / vol_std
    
    return factor
