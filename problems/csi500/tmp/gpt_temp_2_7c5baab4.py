import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate all required components
    for i in range(len(data)):
        if i < 2:  # Need at least 3 days for acceleration calculations
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use data up to current day
        
        # Factor 1: Intraday Momentum Divergence Factor
        # Morning Momentum: (High_t - Open_t) / (Open_t - Low_t)
        morning_momentum = (data['high'].iloc[i] - data['open'].iloc[i]) / max(data['open'].iloc[i] - data['low'].iloc[i], 0.001)
        
        # Afternoon Momentum: (Close_t - Low_t) / (High_t - Close_t)
        afternoon_momentum = (data['close'].iloc[i] - data['low'].iloc[i]) / max(data['high'].iloc[i] - data['close'].iloc[i], 0.001)
        
        # For volume ratios, we assume uniform distribution throughout the day
        # as we don't have intraday volume data
        morning_volume_ratio = 1.0  # Placeholder - would need intraday data
        afternoon_volume_ratio = 1.0  # Placeholder - would need intraday data
        
        # Raw Divergence Score
        divergence_score = (morning_momentum - afternoon_momentum) * (morning_volume_ratio - afternoon_volume_ratio)
        
        # Factor 2: Price-Volume Acceleration Factor
        # Price Change Acceleration
        price_change_accel = (data['close'].iloc[i] - data['close'].iloc[i-1]) - (data['close'].iloc[i-1] - data['close'].iloc[i-2])
        
        # Volume Change Acceleration
        volume_change_accel = (data['volume'].iloc[i] - data['volume'].iloc[i-1]) - (data['volume'].iloc[i-1] - data['volume'].iloc[i-2])
        
        # Combined Acceleration
        combined_accel = price_change_accel * volume_change_accel
        
        # Alignment Multiplier
        alignment_multiplier = 2 if (price_change_accel > 0 and volume_change_accel > 0) or (price_change_accel < 0 and volume_change_accel < 0) else 1
        
        acceleration_factor = combined_accel * alignment_multiplier
        
        # Factor 3: Opening Range Breakout Efficiency
        # For first hour high/low, we use current day's high/low as proxy
        first_hour_high = data['high'].iloc[i]
        first_hour_low = data['low'].iloc[i]
        
        breakout_efficiency = 0
        if data['close'].iloc[i] > first_hour_high:
            breakout_efficiency = (data['close'].iloc[i] - first_hour_high) / max(data['amount'].iloc[i], 0.001)
        elif data['close'].iloc[i] < first_hour_low:
            breakout_efficiency = (first_hour_low - data['close'].iloc[i]) / max(data['amount'].iloc[i], 0.001)
        
        # Volume confirmation
        if i >= 4:
            avg_volume = data['volume'].iloc[i-4:i+1].mean()
            volume_confirmation = data['volume'].iloc[i] / max(avg_volume, 0.001)
        else:
            volume_confirmation = 1.0
        
        breakout_factor = breakout_efficiency * volume_confirmation
        
        # Factor 4: Volatility Compression Expansion Factor
        if i >= 1:
            # Daily Range Ratio
            daily_range_ratio = (data['high'].iloc[i] - data['low'].iloc[i]) / max(data['high'].iloc[i-1] - data['low'].iloc[i-1], 0.001)
            
            # Multi-day Compression
            if i >= 4:
                max_high = data['high'].iloc[i-4:i+1].max()
                min_low = data['low'].iloc[i-4:i+1].min()
                multi_day_compression = (data['high'].iloc[i] - data['low'].iloc[i]) / max(max_high - min_low, 0.001)
            else:
                multi_day_compression = 1.0
            
            # Expansion Magnitude
            expansion_magnitude = abs(data['close'].iloc[i] - data['close'].iloc[i-1]) / max(data['high'].iloc[i] - data['low'].iloc[i], 0.001)
            
            # Volume Expansion
            volume_expansion = data['volume'].iloc[i] / max(data['volume'].iloc[i-1], 0.001)
            
            # Expansion Score
            expansion_score = multi_day_compression * expansion_magnitude * volume_expansion
            
            # Direction from breakout type
            if data['close'].iloc[i] > data['high'].iloc[i-1]:
                expansion_direction = 1
            elif data['close'].iloc[i] < data['low'].iloc[i-1]:
                expansion_direction = -1
            else:
                expansion_direction = 0
            
            volatility_factor = expansion_score * expansion_direction
        else:
            volatility_factor = 0
        
        # Combine all factors with equal weights
        combined_factor = (
            divergence_score + 
            acceleration_factor + 
            breakout_factor + 
            volatility_factor
        ) / 4.0
        
        result.iloc[i] = combined_factor
    
    # Normalize the final result
    if len(result) > 0:
        result = (result - result.mean()) / result.std()
    
    return result
