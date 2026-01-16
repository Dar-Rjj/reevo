import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Price Reversal Component
    low_close_ratio = df['low'] / df['close']
    ema_5_ratio = low_close_ratio.ewm(span=5, adjust=False).mean()
    reversal_signal = low_close_ratio - ema_5_ratio
    
    # Breakout Component
    normalized_range = (df['close'] - df['open']) / df['high']
    normalized_range = np.clip(normalized_range, 0, 1)
    
    # Compute rolling z-score (5-day window)
    mean_5 = normalized_range.rolling(window=5).mean()
    std_5 = normalized_range.rolling(window=5).std()
    breakout_signal = (normalized_range - mean_5) / std_5
    breakout_signal = breakout_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume Trend Strength
    volume_trend = pd.Series(index=df.index, dtype=float)
    for i in range(4, len(df)):
        window = df['volume'].iloc[i-4:i+1]
        if window.min() == window.max():  # Prevent division by zero
            slope = 0
        else:
            slope = linregress(range(5), window.values).slope
        volume_trend.iloc[i] = slope / window.mean()
    
    # Apply sigmoid normalization to volume trend
    volume_strength = 1 / (1 + np.exp(-volume_trend))
    
    # Combine signals
    combined_signal = 0.5 * reversal_signal + 0.5 * breakout_signal
    factor = combined_signal * volume_strength
    
    return factor
