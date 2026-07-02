import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Extremum Signal
    # Calculate rolling rank of close prices over 15 days
    close_rank = df['close'].rolling(window=15, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Calculate 3-day price change and take absolute value
    price_change = df['close'].diff(3).abs()
    
    # Normalize the price change
    normalized_change = price_change / price_change.rolling(window=15, min_periods=1).std()
    
    # Combine to get Price Extremum Signal
    price_signal = close_rank * normalized_change
    
    # Volume Confirmation
    # Calculate ratio of current volume to 15-day rolling mean of volume
    vol_mean = df['volume'].rolling(window=15, min_periods=1).mean()
    vol_ratio = df['volume'] / vol_mean
    
    # Final factor combines price signal with volume confirmation
    factor = np.sign(price_signal) * vol_ratio
    
    return factor
