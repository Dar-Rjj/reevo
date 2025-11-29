import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Efficiency Ratio
    # Price Movement Efficiency
    abs_intraday_return = abs(data['close'] - data['open'])
    total_fluctuation = abs(data['high'] - data['low']) + abs(data['open'] - data['close'].shift(1))
    # Avoid division by zero
    total_fluctuation = total_fluctuation.replace(0, np.nan)
    efficiency_ratio = abs_intraday_return / total_fluctuation
    
    # Volume Confirmation Strength
    volume_5d_avg = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_acceleration = (data['volume'] / volume_5d_avg) - 1
    
    # 3-day exponentially weighted volume ratio
    volume_ewm = volume_acceleration.ewm(span=3, adjust=False).mean()
    
    # Calculate Price-Volume Synchronization
    # Detect directional alignment
    price_direction = np.sign(data['close'] - data['open'])
    volume_direction = np.sign(volume_ewm)
    alignment_score = price_direction * volume_direction
    
    # Compute synchronization strength
    synchronization_strength = abs(efficiency_ratio * alignment_score)
    
    # Generate Composite Factor
    # Combine efficiency and synchronization
    composite_factor = efficiency_ratio * synchronization_strength
    
    # Scale by recent market activity
    abs_price_changes = abs(data['close'] - data['open'])
    market_activity = abs_price_changes.rolling(window=5, min_periods=1).mean()
    # Avoid division by zero
    market_activity = market_activity.replace(0, np.nan)
    scaled_factor = composite_factor / market_activity
    
    # Apply momentum persistence filter
    # Calculate 3-day directional consistency
    price_changes = data['close'] - data['open']
    price_signs = np.sign(price_changes)
    
    # Create consistency calculation
    def calculate_consistency(series):
        if len(series) < 3:
            return np.nan
        current_sign = series.iloc[-1]
        past_signs = series.iloc[-3:-1]  # Last 2 days excluding current
        same_sign_count = (past_signs == current_sign).sum()
        return (same_sign_count + 1) / 3  # +1 for current day
    
    # Calculate rolling consistency
    consistency_ratio = price_signs.rolling(window=3, min_periods=1).apply(
        calculate_consistency, raw=False
    )
    
    # Final factor with momentum persistence
    final_factor = scaled_factor * consistency_ratio
    
    return final_factor
