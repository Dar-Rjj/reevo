import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Intraday Range Component
    df['intraday_range'] = df['high'] - df['low']
    df['normalized_range'] = df['intraday_range'] / df['open']
    
    # Volume Adjustment Component
    df['10d_MA_volume'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['10d_MA_volume']
    df['adjusted_range_efficiency'] = df['normalized_range'] * df['volume_ratio']
    
    # Momentum Scaling Component
    df['intraday_momentum'] = df['open'] - df['close'].shift(1)
    df['momentum_scaled_efficiency'] = df['adjusted_range_efficiency'] * df['intraday_momentum']
    
    # Z-score Normalization
    df['z_score'] = df['momentum_scaled_efficiency'].rolling(window=21, min_periods=1).apply(lambda x: zscore(x)[-1], raw=False)
    
    return df['z_score']
