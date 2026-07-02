import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Prepare necessary columns
    close = df['close']
    volume = df['volume']
    returns = close.pct_change()
    
    # Calculate rolling volatility (20-day std of returns)
    volatility = returns.rolling(window=20).std()
    
    for i in range(len(df)):
        if i < 4:  # Need at least 5 days for calculations
            factor.iloc[i] = 0
            continue
            
        current_idx = df.index[i]
        window_idx = df.index[i-4:i+1]  # 5-day window (t-4 to t)
        
        # Price Trend Component
        # Linear regression slope of close prices (t-4 to t)
        x = np.arange(5)
        y_price = close.loc[window_idx].values
        slope_price = ((x - x.mean()) * (y_price - y_price.mean())).sum() / ((x - x.mean())**2).sum()
        
        # Normalize price slope by volatility
        current_volatility = volatility.loc[current_idx]
        if current_volatility > 0:
            normalized_price_slope = slope_price / current_volatility
        else:
            normalized_price_slope = 0
        
        # Volume Trend Component
        # Linear regression slope of volume (t-4 to t)
        y_volume = volume.loc[window_idx].values
        slope_volume = ((x - x.mean()) * (y_volume - y_volume.mean())).sum() / ((x - x.mean())**2).sum()
        
        # Calculate divergence factor
        if normalized_price_slope != 0:
            sign = np.sign(normalized_price_slope)
            divergence = sign * np.abs(slope_volume - normalized_price_slope)
        else:
            divergence = 0
            
        factor.loc[current_idx] = divergence
    
    return factor
