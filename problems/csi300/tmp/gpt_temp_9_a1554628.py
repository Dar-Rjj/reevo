import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Price Momentum Component
    # Short-term Momentum (5-day)
    df['short_term_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Long-term Mean Reversion (20-day MA)
    df['ma_20_close'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['long_term_reversion'] = (df['close'] - df['ma_20_close']) / df['ma_20_close']
    
    # Volume Confirmation
    # Volume Spike Detection (current volume vs 20-day MA)
    df['ma_20_volume'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_spike'] = df['volume'] / df['ma_20_volume']
    
    # Volume Trend Alignment (5-day slope sign)
    def get_volume_slope(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series)
        return np.sign(slope)
    
    df['volume_trend'] = df['volume'].rolling(window=5, min_periods=1).apply(get_volume_slope, raw=True)
    
    # Signal Combination
    # Multiply Momentum Components
    df['momentum_combo'] = df['short_term_momentum'] * df['long_term_reversion']
    
    # Volume-weighted Final Signal
    df['factor'] = df['momentum_combo'] * df['volume_spike'] * df['volume_trend']
    
    return df['factor']
