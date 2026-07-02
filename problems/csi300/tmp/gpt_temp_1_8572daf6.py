import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Intraday Return with Sign Preservation
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Rolling Z-Score of Intraday Return (3-day window)
    mean_return = df['intraday_return'].rolling(window=3, min_periods=1).mean()
    std_return = df['intraday_return'].rolling(window=3, min_periods=1).std()
    df['rolling_zscore'] = (df['intraday_return'] - mean_return) / std_return
    
    # Volume Spike
    df['20_day_median_volume'] = df['volume'].rolling(window=20, min_periods=1).median()
    df['volume_spike'] = df['volume'] / df['20_day_median_volume']
    
    # Volume Trend (5-day Volume Slope)
    df['volume_trend'] = df['volume'].rolling(window=5).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=True)
    
    # Signal Combination
    df['signal'] = df['rolling_zscore'] * df['volume_spike'] * np.sign(df['volume_trend'])
    
    return df['signal']
