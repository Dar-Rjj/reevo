import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days for some calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Factor 1: Intraday Rejection Momentum Factor
        high_rejection = (current_data['high'].iloc[-1] - current_data['close'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['open'].iloc[-1] + 1e-8)
        low_rejection = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / (current_data['open'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        volume_ratio_1 = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-20:].median()
        rejection_factor = (high_rejection - low_rejection) * volume_ratio_1
        
        # Factor 2: Range Breakout Efficiency Factor
        prior_high = current_data['high'].iloc[-6:-1].max()
        prior_low = current_data['low'].iloc[-6:-1].min()
        prior_range = prior_high - prior_low + 1e-8
        
        if current_data['close'].iloc[-1] > prior_high:
            breakout_pct = (current_data['close'].iloc[-1] - prior_high) / prior_range
            direction = 1
        elif current_data['close'].iloc[-1] < prior_low:
            breakout_pct = (prior_low - current_data['close'].iloc[-1]) / prior_range
            direction = -1
        else:
            breakout_pct = 0
            direction = 0
            
        volume_ratio_2 = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-20:].mean()
        breakout_factor = breakout_pct * volume_ratio_2 * direction
        
        # Factor 3: Momentum Decay Factor
        ret_1d = current_data['close'].iloc[-1] / current_data['close'].iloc[-2] - 1
        ret_3d = current_data['close'].iloc[-1] / current_data['close'].iloc[-4] - 1
        
        if abs(ret_3d) > 1e-8:
            decay_rate = 1 - (ret_1d / ret_3d)
        else:
            decay_rate = 0
            
        # Volume slope (5-day linear regression)
        vol_data = current_data['volume'].iloc[-5:]
        if len(vol_data) >= 2:
            x = np.arange(len(vol_data))
            slope, _, _, _, _ = linregress(x, vol_data)
            volume_slope = slope
        else:
            volume_slope = 0
            
        decay_factor = decay_rate * volume_slope
        
        # Factor 4: Opening Drive Factor
        daily_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8
        
        # For simplicity, using first 30 minutes as opening range (approximation)
        opening_range = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) * 0.3  # Approximation
        sustainability = opening_range / daily_range
        
        # Volume pattern (morning vs afternoon approximation)
        volume_ratio_4 = 1.0  # Simplified - would need intraday data for proper calculation
        
        open_direction = 1 if current_data['close'].iloc[-1] > current_data['open'].iloc[-1] else -1
        opening_factor = sustainability * volume_ratio_4 * open_direction
        
        # Factor 5: Volatility Breakout Factor
        vol_10d = current_data['close'].iloc[-10:].pct_change().std()
        vol_20d = current_data['close'].iloc[-20:].pct_change().std()
        volatility_ratio = vol_10d / (vol_20d + 1e-8)
        
        compressed_range = current_data['high'].iloc[-11:-1].max() - current_data['low'].iloc[-11:-1].min() + 1e-8
        daily_range_current = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        breakout_force = daily_range_current / compressed_range
        
        compressed_volume = current_data['volume'].iloc[-11:-1].mean()
        volume_ratio_5 = current_data['volume'].iloc[-1] / (compressed_volume + 1e-8)
        
        vol_direction = 1 if current_data['close'].iloc[-1] > current_data['open'].iloc[-1] else -1
        volatility_factor = breakout_force * volume_ratio_5 * vol_direction * volatility_ratio
        
        # Combine factors (equal weighting for simplicity)
        combined_factor = (
            rejection_factor + 
            breakout_factor + 
            decay_factor + 
            opening_factor + 
            volatility_factor
        ) / 5
        
        result.iloc[i] = combined_factor
    
    return result
