import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def heuristics_v2(df):
    # Calculate intraday price strength
    intraday_strength = (df['close'] - df['open']) / df['open']
    
    # Calculate rolling 5-day correlation between intraday strength and sequence
    rolling_correlations = pd.Series(index=df.index, dtype=float)
    
    for i in range(4, len(df)):
        if i < 4:  # Not enough data for 5-day window
            rolling_correlations.iloc[i] = np.nan
            continue
        
        window = intraday_strength.iloc[i-4:i+1]  # Current day + previous 4 days
        sequence = np.arange(1, 6)  # [1,2,3,4,5]
        
        # Calculate Pearson correlation using only available data
        if len(window) >= 2:  # Need at least 2 points for correlation
            corr, _ = pearsonr(window, sequence)
            rolling_correlations.iloc[i] = corr
        else:
            rolling_correlations.iloc[i] = np.nan
    
    # Combine intraday strength with trend consistency
    factor = intraday_strength * rolling_correlations
    
    return factor
