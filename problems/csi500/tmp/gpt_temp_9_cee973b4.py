import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate all components
    for i in range(2, len(data)):
        current_date = data.index[i]
        
        # Intraday Momentum Acceleration
        current_high = data['high'].iloc[i]
        current_low = data['low'].iloc[i]
        current_close = data['close'].iloc[i]
        prev_high = data['high'].iloc[i-1]
        prev_low = data['low'].iloc[i-1]
        prev_close = data['close'].iloc[i-1]
        
        current_intraday_range = (current_high - current_low) / current_close
        prev_intraday_range = (prev_high - prev_low) / prev_close
        
        momentum_change = current_intraday_range - prev_intraday_range
        current_volume = data['volume'].iloc[i]
        intraday_momentum_accel = momentum_change * abs(current_volume)
        
        # High-Low Persistence Ratio
        high_streak = 1
        low_streak = 1
        
        # Calculate high persistence
        for j in range(1, min(6, i+1)):
            if data['high'].iloc[i-j] < data['high'].iloc[i-j+1]:
                high_streak += 1
            else:
                break
        
        # Calculate low persistence
        for j in range(1, min(6, i+1)):
            if data['low'].iloc[i-j] < data['low'].iloc[i-j+1]:
                low_streak += 1
            else:
                break
        
        high_persistence = high_streak / 5.0
        low_persistence = low_streak / 5.0
        hl_persistence_ratio = high_persistence * low_persistence
        
        # Opening Gap Volume Efficiency
        current_open = data['open'].iloc[i]
        prev_close_price = data['close'].iloc[i-1]
        opening_gap = (current_open - prev_close_price) / prev_close_price
        
        # Using current day's volume as proxy for opening hour volume
        current_vol = data['volume'].iloc[i]
        avg_volume = data['volume'].iloc[max(0, i-4):i+1].mean()
        volume_ratio = current_vol / avg_volume if avg_volume > 0 else 1
        
        gap_volume_efficiency = opening_gap * volume_ratio
        
        # Price-Volume Divergence Strength
        price_return = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        volume_change = (data['volume'].iloc[i] - data['volume'].iloc[i-1]) / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 0
        
        price_sign = 1 if price_return > 0 else -1
        volume_sign = 1 if volume_change > 0 else -1
        
        if price_sign == volume_sign:
            divergence_strength = price_return * 0.5  # Weak signal
        else:
            divergence_strength = price_return * 2.0  # Strong signal
        
        # Volatility Clustering Intensity
        intraday_vol = (current_high - current_low) / ((current_high + current_low) / 2)
        
        # Calculate recent volatility average (past 5 days)
        recent_vols = []
        for j in range(max(0, i-4), i+1):
            day_high = data['high'].iloc[j]
            day_low = data['low'].iloc[j]
            day_vol = (day_high - day_low) / ((day_high + day_low) / 2)
            recent_vols.append(day_vol)
        
        avg_recent_vol = np.mean(recent_vols) if recent_vols else intraday_vol
        volatility_ratio = intraday_vol / avg_recent_vol if avg_recent_vol > 0 else 1
        volatility_clustering = volatility_ratio * current_vol
        
        # Close-to-Open Reversal Momentum
        prev_open = data['open'].iloc[i-1]
        prev_close_prev = data['close'].iloc[i-1]
        prev_reversal = (prev_open - prev_close_prev) / prev_close_prev
        
        current_reversal = (current_open - prev_close_price) / prev_close_price
        volume_change_rev = (current_vol - data['volume'].iloc[i-1]) / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 0
        
        reversal_momentum = current_reversal * volume_change_rev
        
        # Combine all factors with equal weights
        combined_factor = (
            intraday_momentum_accel +
            hl_persistence_ratio +
            gap_volume_efficiency +
            divergence_strength +
            volatility_clustering +
            reversal_momentum
        )
        
        factor_values.iloc[i] = combined_factor
    
    # Fill NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
