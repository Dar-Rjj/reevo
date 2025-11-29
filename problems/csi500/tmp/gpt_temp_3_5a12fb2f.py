import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range components
    high_low_range = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    
    # True Range is the maximum of the three components
    true_range = pd.concat([high_low_range, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Calculate opening gap
    opening_gap = abs(df['open'] - df['close'].shift(1))
    
    # Normalize True Range by opening gap (avoid division by zero)
    normalized_trend_strength = true_range / (opening_gap + 1e-8)
    
    # Calculate volume ratio
    rolling_median_volume = df['volume'].rolling(window=5, min_periods=1).median()
    volume_ratio = df['volume'] / (rolling_median_volume + 1e-8)
    
    # Combine trend strength with volume confirmation
    trend_strength = normalized_trend_strength * volume_ratio
    
    # Calculate High-Low midpoint
    hl_midpoint = (df['high'] + df['low']) / 2
    
    # Calculate rolling correlation between midpoint and time index
    time_index = np.arange(len(df))
    correlation_values = []
    
    for i in range(len(df)):
        if i < 4:  # Not enough data for 5-day window
            correlation_values.append(0)
        else:
            window_midpoints = hl_midpoint.iloc[i-4:i+1]
            window_time = time_index[i-4:i+1]
            corr = np.corrcoef(window_midpoints, window_time)[0, 1]
            correlation_values.append(corr if not np.isnan(corr) else 0)
    
    trend_persistence = pd.Series(correlation_values, index=df.index)
    
    # Calculate recent volatility (10-day standard deviation of returns)
    returns = df['close'].pct_change()
    recent_volatility = returns.rolling(window=10, min_periods=1).std()
    
    # Weight persistence by volatility
    weighted_persistence = trend_persistence * recent_volatility
    
    # Combine components
    combined_factor = trend_strength * weighted_persistence
    
    # Apply sign adjustment based on intraday direction
    sign_adjustment = np.where(df['close'] > df['open'], 1, -1)
    final_factor = combined_factor * sign_adjustment
    
    return final_factor
