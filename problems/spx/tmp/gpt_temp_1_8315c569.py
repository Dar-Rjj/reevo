import pandas as pd
import numpy as np
def heuristics_v2(df):
    """
    Calculate Price Momentum with Volume Confirmation factor.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with columns ['close', 'volume'] and datetime index
    
    Returns:
    pd.Series: Factor values indexed by date
    """
    # Calculate Price Momentum (rolling window of 20 days)
    momentum = df['close'].pct_change(periods=20)
    
    # Calculate Volume Trend Strength (linear regression slope over 20 days)
    def volume_slope(vol_series):
        from scipy.stats import linregress
        x = np.arange(len(vol_series))
        slope = linregress(x, vol_series.values).slope
        return slope
    
    volume_trend = df['volume'].rolling(window=20).apply(volume_slope, raw=False)
    
    # Combine momentum with volume confirmation
    factor = momentum * volume_trend
    
    return factor
