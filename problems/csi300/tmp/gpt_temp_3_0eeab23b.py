import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Price Momentum (PM)
    # Rolling Mean Advance Rate (RMAR)
    df['Price_Change'] = df['close'] - df['close'].shift(2)
    df['RMAR'] = df['Price_Change'].rolling(window=10, min_periods=1).mean()
    
    # Normalize RMAR over asset pool
    df['RMAR_Normalized'] = (df['RMAR'] - df['RMAR'].mean()) / df['RMAR'].std()
    
    # Positive momentum triggers recognition: RMAR >= Threshold
    threshold = df['RMAR'].quantile(0.75)
    df['PM_Signal'] = np.where(df['RMAR'] >= threshold, 1, 0)
    
    # Velocity Amplitude (VA)
    # Historical Velocity Extension (HVE)
    df['PES'] = (df['high'] - df['low']) / df['open']
    df['PES_ZScore'] = df['PES'].rolling(window=3, min_periods=1).apply(lambda x: zscore(x)[-1])
    
    # High-Low Momentum Price Flow (HLMPF)
    df['HLMPF'] = (df['high'] - df['low']) / df['open'].shift(1)
    
    # Final Factor Construction
    # Composite PM (MSV)
    df['MSV'] = df['RMAR_Normalized']
    
    # Adaptive Strategy
    df['Adaptive_Factor'] = 0.5 * df['MSV'] + 0.5 * df['HLMPF']
    
    return df['Adaptive_Factor']
