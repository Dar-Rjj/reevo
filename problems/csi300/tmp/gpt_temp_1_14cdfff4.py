import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    def calculate_slope(series, window=5):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            X = np.arange(window).reshape(-1, 1)
            y = series.iloc[i-window+1:i+1].values
            model = LinearRegression()
            model.fit(X, y)
            slopes.iloc[i] = model.coef_[0]
        return slopes

    # Calculate 5-day Price Slope
    price_slope = calculate_slope(df['close'])
    
    # Calculate 5-day Volume Slope
    volume_slope = calculate_slope(df['volume'])
    
    # Normalize Components
    normalized_price_slope = price_slope / df['close']
    normalized_volume_slope = volume_slope / df['volume']
    
    # Compute Divergence Signal
    divergence_signal = -normalized_price_slope + normalized_volume_slope
    
    return divergence_signal
