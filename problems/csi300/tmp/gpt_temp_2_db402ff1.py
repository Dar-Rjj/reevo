import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate intraday price deviation
    df['intraday_dev'] = (df['high'] - df['low']) / df['open']
    
    # Calculate 5-day rolling standard deviation of intraday deviation
    df['intraday_vol'] = df['intraday_dev'].rolling(5, min_periods=1).std()
    
    # Calculate closing price efficiency
    midpoint = (df['high'] + df['low']) / 2
    df['close_efficiency'] = abs(df['close'] - midpoint) / (df['high'] - df['low'])
    df['close_efficiency'] = df['close_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate 5-day rolling mean of closing efficiency
    df['eff_rolling'] = df['close_efficiency'].rolling(5, min_periods=1).mean()
    
    # Combine to get price inefficiency measure
    df['price_inefficiency'] = df['intraday_vol'] * df['eff_rolling']
    
    # Calculate volume directionality
    def linear_reg_slope(x):
        if len(x) < 2:
            return 0
        x = np.arange(len(x))
        y = x.copy()
        mask = ~np.isnan(x) & ~np.isnan(y)
        if sum(mask) < 2:
            return 0
        slope = np.polyfit(x[mask], y[mask], 1)[0]
        return slope
    
    df['volume_slope'] = df['volume'].rolling(3, min_periods=1).apply(linear_reg_slope, raw=False)
    df['volume_dir'] = np.sign(df['volume_slope'])
    
    # Final factor construction
    factor = df['price_inefficiency'] * df['volume_dir']
    
    return factor
