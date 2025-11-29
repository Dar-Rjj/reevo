import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate intraday reversal components
    # Morning Reversal Signal: (Open - Low of first 30 minutes) / (High of first 30 minutes - Low of first 30 minutes)
    # Since we only have daily OHLC, we'll approximate using the daily range
    morning_reversal = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Afternoon Momentum Decay: (Close - High of last hour) / (High of last hour - Low of last hour)
    # Using daily close and high/low range
    afternoon_decay = (data['close'] - data['high']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume Acceleration Dynamics
    # Calculate 5-hour (5-day) moving average of volume
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Volume surge detection: current volume vs 5-period moving average
    volume_surge = data['volume'] / (volume_ma_5 + 1e-8)
    
    # Volume acceleration rate: current/previous hour ratio
    volume_acceleration = data['volume'] / (data['volume'].shift(1) + 1e-8)
    
    # Price-Volume Divergence
    # Combine reversal signals
    combined_reversal = morning_reversal + afternoon_decay
    
    # Calculate divergence between price reversal and volume acceleration
    price_volume_divergence = combined_reversal * volume_acceleration
    
    # Multi-Timeframe Confirmation
    # 2-hour (2-day) rolling correlation between reversal signals and volume changes
    reversal_volume_corr = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 1:  # Need at least 2 periods for correlation
            start_idx = max(0, i-1)
            window_data = data.iloc[start_idx:i+1]
            if len(window_data) >= 2:
                reversal_window = combined_reversal.iloc[start_idx:i+1]
                volume_window = data['volume'].iloc[start_idx:i+1]
                if reversal_window.std() > 0 and volume_window.std() > 0:
                    reversal_volume_corr.iloc[i] = reversal_window.corr(volume_window)
                else:
                    reversal_volume_corr.iloc[i] = 0
            else:
                reversal_volume_corr.iloc[i] = 0
        else:
            reversal_volume_corr.iloc[i] = 0
    
    # Intraday pattern consistency across 3 consecutive hours (days)
    pattern_consistency = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 2:  # Need at least 3 periods for consistency check
            window_data = data.iloc[i-2:i+1]
            reversal_window = combined_reversal.iloc[i-2:i+1]
            volume_window = volume_acceleration.iloc[i-2:i+1]
            
            # Check if reversal signals have consistent direction
            reversal_trend = np.sign(reversal_window.diff().sum())
            volume_trend = np.sign(volume_window.diff().sum())
            
            # Consistency score: higher when both trends align
            pattern_consistency.iloc[i] = 1 if reversal_trend == volume_trend else -1
        else:
            pattern_consistency.iloc[i] = 0
    
    # Combine all components into final factor
    factor_values = (
        morning_reversal * 0.3 +
        afternoon_decay * 0.3 +
        volume_surge * 0.15 +
        price_volume_divergence * 0.15 +
        reversal_volume_corr * 0.05 +
        pattern_consistency * 0.05
    )
    
    # Normalize the factor
    factor_values = (factor_values - factor_values.rolling(window=20, min_periods=10).mean()) / (factor_values.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return factor_values
