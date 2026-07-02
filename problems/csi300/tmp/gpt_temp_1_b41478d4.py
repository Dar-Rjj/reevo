import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Calculate price slope (5-day linear regression)
    def linear_reg_slope(series, window=5):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                slope = ((window * (x*y).sum()) - (x.sum() * y.sum())) / \
                        (window * (x**2).sum() - (x.sum())**2)
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    price_slope = linear_reg_slope(data['close'], window=5)
    
    # Calculate price range normalization
    price_range = (data['high'] - data['low']) / data['close']
    normalized_price_slope = price_slope * price_range
    
    # Calculate volume slope (5-day linear regression)
    volume_slope = linear_reg_slope(data['volume'], window=5)
    
    # Create divergence signal
    divergence_signal = np.where(
        ((normalized_price_slope > 0) & (volume_slope < 0)) | 
        ((normalized_price_slope < 0) & (volume_slope > 0)),
        normalized_price_slope * -volume_slope,
        0
    )
    
    # Calculate 20-day volatility (rolling std)
    volatility = data['close'].rolling(window=20, min_periods=1).std()
    
    # Final factor calculation
    factor = divergence_signal / volatility
    
    return pd.Series(factor, index=data.index)
