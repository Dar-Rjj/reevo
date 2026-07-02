import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Price Component
    close = df['close']
    
    # Calculate 5-day linear regression slope for price trend
    price_trend = close.rolling(window=5).apply(
        lambda x: linregress(np.arange(5), x)[0], raw=True
    )
    # Normalize price trend
    normalized_price_trend = (price_trend / close) * 100
    
    # Volume Component
    volume = df['volume']
    
    # Calculate 5-day EMA slope for volume trend
    ema_volume = volume.ewm(span=5, adjust=False).mean()
    volume_trend = ema_volume.diff()
    # Normalize volume trend
    normalized_volume_trend = (volume_trend / volume) * 100
    
    # Divergence Signal
    signal = pd.Series(0, index=df.index)
    
    # Positive signal when price down and volume up
    signal[(price_trend < 0) & (volume_trend > 0)] = 1
    
    # Negative signal when price up and volume down
    signal[(price_trend > 0) & (volume_trend < 0)] = -1
    
    # Price-weighted magnitude
    price_std = close.rolling(window=20).std()
    factor = signal * normalized_price_trend / price_std
    
    return factor
