import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Price Slope Calculation
    def calculate_price_slope(close_prices):
        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    # Volume Slope Calculation
    def calculate_volume_slope(volumes):
        X = np.arange(len(volumes)).reshape(-1, 1)
        y = volumes.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    # Initialize result Series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        # Calculate Price Slope (past 5 days)
        close_prices = df['close'].iloc[i-5:i]
        price_slope = calculate_price_slope(close_prices)
        
        # Calculate Volume Slope (past 5 days)
        volumes = df['volume'].iloc[i-5:i]
        volume_slope = calculate_volume_slope(volumes)
        
        # Compute Divergence
        divergence = price_slope * volume_slope
        
        # Sign adjustment
        if price_slope > 0 and volume_slope < 0:
            divergence = abs(divergence)  # Strong signal
        
        factor_values.iloc[i] = divergence
    
    return factor_values
