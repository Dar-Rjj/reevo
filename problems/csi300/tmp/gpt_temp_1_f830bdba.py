import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Price Reversal Component
    df['early_strength'] = (df['high'] - df['open']) / df['open']
    df['late_weakness'] = (df['close'] - df['low']) / df['low']
    
    # Volume Confirmation
    df['volume_surge'] = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_threshold'] = np.where(df['volume_surge'] > 1.5, 1, 0)
    
    # Combined Factor
    df['reversal_signal'] = df['early_strength'] - df['late_weakness']
    df['combined_factor'] = df['reversal_signal'] * df['volume_surge'] * df['volume_threshold']
    
    # Normalize Factor
    df['normalized_factor'] = df['combined_factor'].rolling(window=10, min_periods=1).apply(lambda x: zscore(x)[-1])
    
    return df['normalized_factor']
