import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Function to calculate linear regression slope
    def calculate_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            X = np.arange(window).reshape(-1, 1)
            y = series.iloc[i-window+1:i+1].values
            model = LinearRegression()
            model.fit(X, y)
            slopes.iloc[i] = model.coef_[0]
        return slopes
    
    # Calculate 5-day Price Slope using Close price
    price_slope = calculate_slope(df['close'], window=5)
    
    # Calculate 5-day Volume Slope using Volume
    volume_slope = calculate_slope(df['volume'], window=5)
    
    # Compute Divergence Signal
    divergence_signal = price_slope * volume_slope
    
    # Take inverse when signs differ
    factor = divergence_signal.copy()
    factor[(price_slope > 0) & (volume_slope < 0)] *= -1
    factor[(price_slope < 0) & (volume_slope > 0)] *= -1
    
    return factor
