import pandas as pd
import numpy as np
def heuristics_v2(df):
    """
    Price-Volume Divergence Factor that combines price trend strength and volume trend divergence.
    
    Parameters:
    df (pd.DataFrame): DataFrame with market data (open, high, low, close, amount, volume)
    
    Returns:
    pd.Series: Factor values indexed by date
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # Initialize result Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Rolling window size
    window = 5
    
    for i in range(window-1, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Get the last window days of data
        window_data = current_data.iloc[-window:]
        
        # 1. Price Trend Component
        # Linear regression of close prices
        X = np.arange(window).reshape(-1, 1)
        y_price = window_data['close'].values
        reg_price = LinearRegression().fit(X, y_price)
        price_slope = reg_price.coef_[0]
        
        # Normalize by price level
        current_close = window_data['close'].iloc[-1]
        normalized_price_slope = price_slope / current_close
        
        # 2. Volume Trend Divergence
        # Linear regression of volume
        y_volume = window_data['volume'].values
        reg_volume = LinearRegression().fit(X, y_volume)
        volume_slope = reg_volume.coef_[0]
        
        # Compute divergence score
        divergence = price_slope - volume_slope
        
        # Scale by price volatility
        price_std = window_data['close'].std()
        if price_std > 0:
            scaled_divergence = divergence / price_std
        else:
            scaled_divergence = 0
        
        # Combine components
        factor_value = normalized_price_slope + scaled_divergence
        factor.iloc[i] = factor_value
    
    return factor
