import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(data):
    # Compute Momentum
    rolling_mean = data['close'].rolling(window=5, min_periods=1).mean()
    momentum = data['close'] / rolling_mean
    
    # Initialize output Series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate for each day
    for i in range(2, len(data)):
        # Current date
        t = data.index[i]
        
        # Get last 3 days data (including current day t)
        window_close = data['close'].iloc[i-2:i+1].values.reshape(-1, 1)
        window_momentum = momentum.iloc[i-2:i+1].values.reshape(-1, 1)
        X = np.array([0, 1, 2]).reshape(-1, 1)  # Time points
        
        # Compute Current Trend Strength (R-squared)
        model_close = LinearRegression().fit(X, window_close)
        ss_res_close = np.sum((window_close - model_close.predict(X))**2)
        ss_tot_close = np.sum((window_close - np.mean(window_close))**2)
        r2_close = 1 - (ss_res_close / ss_tot_close) if ss_tot_close != 0 else 0
        
        # Compute Momentum Trend Strength (R-squared)
        model_momentum = LinearRegression().fit(X, window_momentum)
        ss_res_momentum = np.sum((window_momentum - model_momentum.predict(X))**2)
        ss_tot_momentum = np.sum((window_momentum - np.mean(window_momentum))**2)
        r2_momentum = 1 - (ss_res_momentum / ss_tot_momentum) if ss_tot_momentum != 0 else 0
        
        # Compare Trend Strengths
        factor_values[t] = r2_close - r2_momentum
    
    return factor_values
