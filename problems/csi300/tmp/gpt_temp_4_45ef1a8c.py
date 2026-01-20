import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Momentum Component
    # Price Acceleration
    sma5 = df['close'].rolling(5, min_periods=1).mean()
    sma10 = df['close'].rolling(10, min_periods=1).mean()
    std20_close = df['close'].rolling(20, min_periods=1).std()
    price_acceleration = (sma5 - sma10) / (std20_close + 1e-6)
    
    # Intraday Momentum
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-6)
    std5_intraday = (df['close'] - df['open']).rolling(5, min_periods=1).std()
    intraday_momentum = intraday_momentum / (std5_intraday + 1e-6)
    
    # Combine momentum components
    momentum = price_acceleration + intraday_momentum
    
    # Volume Confirmation
    # Volume Spike Detection
    sma20_volume = df['volume'].rolling(20, min_periods=1).mean()
    volume_ratio = df['volume'] / (sma20_volume + 1e-6)
    volume_spike = np.where(volume_ratio > 1, np.log(volume_ratio), -1/volume_ratio)
    
    # Volume Weighting
    def rolling_percentile(s):
        return s.rolling(20, min_periods=1).apply(lambda x: (x[-1] > x[:-1]).mean() * 100)
    
    volume_percentile = rolling_percentile(df['volume'])
    volume_weight = np.exp(3 * volume_percentile / 100)
    
    # Combine volume components
    volume_confirmation = volume_spike * volume_weight
    
    # Signal Combination
    signal = momentum * volume_confirmation
    
    # Final Z-Score normalization
    factor = (signal - signal.rolling(5, min_periods=1).mean()) / (signal.rolling(5, min_periods=1).std() + 1e-6)
    
    return factor
