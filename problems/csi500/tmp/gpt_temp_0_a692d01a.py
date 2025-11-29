import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    prev_close = data['close'].shift(1)
    high_low = data['high'] - data['low']
    high_prev_close = abs(data['high'] - prev_close)
    low_prev_close = abs(data['low'] - prev_close)
    data['true_range'] = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    
    # Calculate 10-day Average True Range
    data['atr_10'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    
    # Calculate Volatility Ratio
    data['vol_ratio'] = data['true_range'] / data['atr_10'].replace(0, 1)
    
    # Calculate intraday momentum slope (using hourly data simulation)
    # Since we don't have true hourly data, we'll simulate using rolling windows
    # Assuming each day has 6.5 trading hours, we'll create 6 intraday periods
    data['price_change_1'] = data['close'].pct_change(periods=1)
    data['price_change_2'] = data['close'].pct_change(periods=2)
    data['price_change_3'] = data['close'].pct_change(periods=3)
    data['price_change_4'] = data['close'].pct_change(periods=4)
    
    # Calculate momentum slope using linear regression on last 4 periods
    slopes = []
    for i in range(len(data)):
        if i < 3:
            slopes.append(0)
            continue
        
        # Use last 4 price changes as "hourly" data points
        X = np.array([1, 2, 3, 4]).reshape(-1, 1)
        y = np.array([
            data['price_change_1'].iloc[i],
            data['price_change_2'].iloc[i],
            data['price_change_3'].iloc[i],
            data['price_change_4'].iloc[i]
        ])
        
        if np.all(np.isfinite(y)):
            model = LinearRegression()
            model.fit(X, y)
            slopes.append(model.coef_[0])
        else:
            slopes.append(0)
    
    data['momentum_slope'] = slopes
    
    # Calculate acceleration breakpoints
    data['slope_std_3'] = data['momentum_slope'].rolling(window=3, min_periods=1).std()
    data['acceleration_flag'] = (data['momentum_slope'] > 2 * data['slope_std_3'].replace(0, 1)).astype(int)
    
    # Volume divergence analysis
    # Calculate volume percentage relative to daily average
    data['volume_pct'] = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    
    # Calculate price move magnitude
    data['price_move'] = abs(data['close'].pct_change())
    
    # Detect volume-price divergence
    # High volume but small price moves indicate divergence
    data['volume_divergence'] = np.where(
        (data['volume_pct'] > 1.2) & (data['price_move'] < 0.01),
        data['volume_pct'] * (0.01 / (data['price_move'] + 0.0001)),
        1.0
    )
    
    # Combine volatility and acceleration signals
    data['vol_accel_signal'] = data['vol_ratio'] * data['momentum_slope'] * data['acceleration_flag']
    
    # Apply exponential weighting (simulating recent hours)
    # Using exponential moving average to give more weight to recent periods
    data['weighted_signal'] = data['vol_accel_signal'].ewm(span=2, adjust=False).mean()
    
    # Integrate volume divergence
    data['final_signal'] = data['weighted_signal'] * data['volume_divergence']
    
    # Apply directional adjustment based on price movement direction
    price_direction = np.sign(data['close'].pct_change())
    data['final_factor'] = data['final_signal'] * price_direction.replace(0, 1)
    
    # Return the factor series
    return data['final_factor']
