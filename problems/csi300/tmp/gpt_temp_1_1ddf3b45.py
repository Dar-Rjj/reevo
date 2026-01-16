import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate momentum components
    df['short_term_momentum'] = df['close'] / df['close'].shift(5) - 1
    df['long_term_momentum'] = df['close'] / df['close'].shift(20) - 1
    df['momentum_divergence'] = df['short_term_momentum'] - df['long_term_momentum']
    
    # Calculate volume slope using rolling linear regression
    def calculate_volume_slope(series):
        if len(series) < 2:  # Need at least 2 points for regression
            return np.nan
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0][0]
    
    # Apply rolling volume slope calculation (10-day window)
    df['volume_slope'] = df['volume'].rolling(window=10, min_periods=2).apply(calculate_volume_slope, raw=False)
    
    # Combine signals
    factor = df['momentum_divergence'] * df['volume_slope']
    
    return factor
