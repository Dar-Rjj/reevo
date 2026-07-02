import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

def heuristics_v2(df):
    # Calculate Normalized Intraday Range
    df['RawRange'] = df['high'] - df['low']
    df['NormalizedRange'] = df['RawRange'] / df['close'].shift(1)
    
    # Calculate Short-Term Momentum
    df['Momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Calculate Historical Range Percentile
    df['RangePercentile'] = df['NormalizedRange'].rolling(window=20, min_periods=1).apply(lambda x: percentileofscore(x, x[-1]))
    
    # Create Weighted Composite
    df['Factor'] = df['RangePercentile'] * df['Momentum']
    
    # Return the factor series
    return df['Factor']
