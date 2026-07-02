import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import rankdata

def heuristics_v2(data):
    df = data.copy()
    
    # Compute Normalized Intraday Range
    # Raw High-Low Range
    df['raw_range'] = df['high'] - df['low']
    
    # 20-day Std Dev of Close
    df['close_std_20d'] = df['close'].rolling(20).std()
    
    # 5-day rolling percentile of Volume
    df['volume_rank'] = df['volume'].rolling(5).apply(lambda x: rankdata(x)[-1]/len(x) if len(x) == 5 else np.nan)
    
    # Adjust for Volatility
    df['norm_range'] = (df['raw_range'] / df['close_std_20d']) * df['volume_rank']
    
    # Generate Reversal Signal
    # Midpoint Deviation
    df['midpoint'] = (df['high'] + df['low']) / 2
    df['midpoint_dev'] = (df['close'] - df['midpoint']) / df['norm_range'].replace(0, np.nan)
    
    # 5-day Return
    df['ret_5d'] = df['close'].pct_change(5)
    
    # Volume Weight
    df['log_volume'] = np.log(df['volume'])
    df['avg_volume_5d'] = df['volume'].rolling(5).mean()
    df['volume_weight'] = df['log_volume'] * (df['volume'] / df['avg_volume_5d'])
    
    # Combine with Trend Direction
    df['reversal_signal'] = df['midpoint_dev'] * df['ret_5d'] * df['volume_weight']
    
    # Final Adjustment
    # Winsorize at 5th/95th percentiles
    lower = df['reversal_signal'].quantile(0.05)
    upper = df['reversal_signal'].quantile(0.95)
    df['factor'] = df['reversal_signal'].clip(lower=lower, upper=upper)
    
    # Standardize
    mean = df['factor'].mean()
    std = df['factor'].std()
    df['factor'] = (df['factor'] - mean) / std
    
    return df['factor']
