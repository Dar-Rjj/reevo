import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Initialize output series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling linear regression slopes for price and volume
    for i in range(len(df)):
        if i < 4:  # Need at least 5 days for calculation
            factor.iloc[i] = 0
            continue
            
        # Get past 5 days data (including current day)
        window = df.iloc[i-4:i+1]
        
        # Price slope calculation
        x = np.arange(5)
        y_price = window['close'].values
        price_slope = np.cov(x, y_price)[0,1] / np.var(x)
        
        # Volume slope calculation
        y_volume = window['volume'].values
        volume_slope = np.cov(x, y_volume)[0,1] / np.var(x)
        
        # Calculate divergence score
        divergence_score = np.sign(price_slope * volume_slope)
        factor.iloc[i] = divergence_score
    
    return factor
