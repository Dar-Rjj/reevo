import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic components for each day
    for i in range(len(data)):
        if i < 20:  # Need enough history for calculations
            factor.iloc[i] = 0
            continue
            
        current_day = data.iloc[i]
        
        # Morning Session Components
        morning_high = current_day['high']  # Using daily high as proxy for first hour high
        morning_low = current_day['low']    # Using daily low as proxy for first hour low
        open_price = current_day['open']
        close_price = current_day['close']
        
        # First hour price efficiency
        first_hour_efficiency = (morning_high - open_price) / open_price if open_price != 0 else 0
        
        # Morning volume concentration (using daily volume as proxy)
        current_volume = current_day['volume']
        prev_volumes = data['volume'].iloc[max(0, i-5):i]  # 5-day average
        avg_prev_volume = prev_volumes.mean() if len(prev_volumes) > 0 else current_volume
        morning_volume_ratio = current_volume / avg_prev_volume if avg_prev_volume != 0 else 1
        
        # Afternoon Session Components
        afternoon_low = morning_low  # Using daily low as proxy
        afternoon_efficiency = (close_price - afternoon_low) / close_price if close_price != 0 else 0
        
        # Afternoon volume persistence (using same volume)
        afternoon_volume_ratio = 1.0  # Simplified assumption
        
        # Intraday Volatility Structure
        morning_range = morning_high - morning_low if morning_low != 0 else 0
        afternoon_range = morning_range  # Simplified assumption
        
        # Regime Persistence Signal
        volatility_ratio = afternoon_range / morning_range if morning_range != 0 else 1
        regime_persistence = (first_hour_efficiency * afternoon_efficiency * 
                            morning_volume_ratio) / max(volatility_ratio, 0.1)
        
        # Session Alignment
        morning_direction = 1 if morning_high > open_price else -1
        afternoon_direction = 1 if close_price > afternoon_low else -1
        session_consistency = 1 if morning_direction == afternoon_direction else 0.5
        
        # Volume-Weighted Acceleration Components
        # Short-term volume acceleration
        vol_3ma = data['volume'].iloc[i-2:i+1].mean()
        vol_deviation = current_volume / vol_3ma if vol_3ma != 0 else 1
        vol_acceleration = vol_deviation - 1
        
        # Long-term volume trend
        vol_10ma = data['volume'].iloc[i-9:i+1].mean()
        vol_20ma = data['volume'].iloc[i-19:i+1].mean()
        vol_trend_consistency = vol_10ma / vol_20ma if vol_20ma != 0 else 1
        
        # Price acceleration components
        price_3d_change = (close_price / data['close'].iloc[i-2] - 1) if i >= 2 else 0
        price_5d_change = (close_price / data['close'].iloc[i-4] - 1) if i >= 4 else 0
        price_acceleration = price_3d_change - price_5d_change
        
        # Volume-Price Divergence Signal
        divergence_signal = (vol_acceleration * price_acceleration * 
                           vol_trend_consistency * session_consistency)
        
        # Range Expansion Components
        current_range = morning_high - morning_low
        prev_range = data['high'].iloc[i-1] - data['low'].iloc[i-1]
        range_expansion = current_range / prev_range if prev_range != 0 else 1
        
        range_5ma = pd.Series([data['high'].iloc[j] - data['low'].iloc[j] 
                              for j in range(max(0, i-4), i+1)]).mean()
        range_expansion_magnitude = current_range / range_5ma if range_5ma != 0 else 1
        
        # Volume expansion
        vol_5ma = data['volume'].iloc[max(0, i-4):i+1].mean()
        volume_expansion = current_volume / vol_5ma if vol_5ma != 0 else 1
        
        # Range Momentum Signal
        range_momentum = (range_expansion * volume_expansion * 
                         range_expansion_magnitude * afternoon_direction)
        
        # Price-Level Concentration Components
        morning_pivot = (morning_high + morning_low) / 2
        afternoon_pivot = morning_pivot  # Simplified assumption
        
        # Level concentration (simplified)
        level_concentration = 1 - (abs(close_price - morning_pivot) / morning_pivot) if morning_pivot != 0 else 0
        
        # Volume concentration ratio
        volume_concentration_ratio = 1.0  # Simplified
        
        # Level Momentum Signal
        level_momentum = level_concentration * volume_concentration_ratio * afternoon_direction
        
        # Price Efficiency Components
        open_close_efficiency = abs(close_price - open_price) / open_price if open_price != 0 else 0
        morning_efficiency = first_hour_efficiency
        afternoon_session_efficiency = afternoon_efficiency
        
        # Multi-timeframe momentum
        mom_2d = (close_price / data['close'].iloc[i-1] - 1) if i >= 1 else 0
        mom_5d = price_5d_change
        momentum_persistence = abs(mom_2d) / (abs(mom_5d) + 0.001)
        
        # Efficiency-Momentum Signal
        efficiency_momentum = (open_close_efficiency * momentum_persistence * 
                              morning_efficiency * afternoon_session_efficiency)
        
        # Combine all components with appropriate weights
        final_factor = (regime_persistence * 0.25 + 
                       divergence_signal * 0.20 + 
                       range_momentum * 0.20 + 
                       level_momentum * 0.15 + 
                       efficiency_momentum * 0.20)
        
        factor.iloc[i] = final_factor
    
    # Normalize the factor
    if len(factor) > 0:
        factor = (factor - factor.mean()) / (factor.std() + 1e-8)
    
    return factor
