import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Initialize output Series
    signal = pd.Series(index=df.index, dtype=float)
    
    # Price Reversal Component
    low_close_ratio = df['low'] / df['close']
    ema_5_ratio = low_close_ratio.ewm(span=5, adjust=False).mean()
    reversal_signal = low_close_ratio - ema_5_ratio
    
    # Breakout Component
    normalized_range = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    normalized_range = np.clip(normalized_range, -1, 1)
    rolling_mean = normalized_range.rolling(window=5).mean()
    rolling_std = normalized_range.rolling(window=5).std()
    breakout_signal = (normalized_range - rolling_mean) / (rolling_std + 1e-8)
    
    # Volume Trend Strength
    volume = df['volume']
    def get_slope(x):
        if len(x) < 2:
            return np.nan
        return linregress(np.arange(len(x)), x).slope
    
    volume_slope = volume.rolling(window=5).apply(get_slope, raw=True)
    volume_mean = volume.rolling(window=5).mean()
    volume_trend = volume_slope / (volume_mean + 1e-8)
    volume_strength = 1 / (1 + np.exp(-volume_trend))  # Sigmoid normalization
    
    # Final Signal Combination
    momentum_signal = (reversal_signal + breakout_signal) / 2
    signal = momentum_signal * volume_strength
    
    return signal
