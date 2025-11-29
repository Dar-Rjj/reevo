import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Create copies to avoid SettingWithCopyWarning
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need enough data for rolling calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Price Compression Dynamics
        # For simplicity, we'll approximate intraday patterns using daily data
        # In practice, you would need actual intraday data for these calculations
        morning_compression = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / current_data['open'].iloc[-1]
        afternoon_expansion = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / current_data['high'].iloc[-1]
        daily_compression_ratio = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / (morning_compression + afternoon_expansion + 1e-8)
        
        # Position-Based Acceleration
        opening_acceleration = (current_data['high'].iloc[-1] - current_data['open'].iloc[-1]) / (current_data['open'].iloc[-1] + 1e-8)
        closing_momentum = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / (current_data['low'].iloc[-1] + 1e-8)
        position_bias = ((current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / (current_data['close'].iloc[-1] + 1e-8) - 
                        (current_data['high'].iloc[-1] - current_data['close'].iloc[-1]) / (current_data['close'].iloc[-1] + 1e-8))
        
        # Volume Divergence Patterns
        # Using daily volume as approximation since we don't have intraday volume data
        volume_decay_ratio = 1.0  # Placeholder - would need intraday volume data
        concentration_shift = 1.0  # Placeholder - would need intraday volume data
        
        # Calculate volume acceleration using available data
        if i >= 2:
            volume_acceleration = (current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2]) / (current_data['volume'].iloc[-2] + 1e-8)
            price_change_2d = current_data['close'].iloc[-1] / current_data['close'].iloc[-3] - 1
            volume_price_divergence = volume_acceleration / (price_change_2d + 1e-8)
        else:
            volume_price_divergence = 0
        
        # Shadow Rejection Analysis
        upper_shadow_rejection = (current_data['high'].iloc[-1] - max(current_data['open'].iloc[-1], current_data['close'].iloc[-1])) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        lower_shadow_rejection = (min(current_data['open'].iloc[-1], current_data['close'].iloc[-1]) - current_data['low'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8)
        net_rejection = upper_shadow_rejection - lower_shadow_rejection
        
        # Multi-Timeframe Momentum
        if i >= 8:
            short_term_trend = current_data['close'].iloc[-1] / current_data['close'].iloc[-4] - 1
            medium_term_trend = current_data['close'].iloc[-1] / current_data['close'].iloc[-9] - 1
            trend_consistency = short_term_trend / (medium_term_trend + 1e-8)
        else:
            trend_consistency = 0
        
        # Volatility Context
        intraday_volatility_range = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / (current_data['close'].iloc[-1] + 1e-8)
        
        # Calculate rolling volatility mean
        recent_volatility = []
        for j in range(max(0, i-19), i+1):
            vol_range = (current_data['high'].iloc[j] - current_data['low'].iloc[j]) / (current_data['close'].iloc[j] + 1e-8)
            recent_volatility.append(vol_range)
        
        volatility_adjustment = intraday_volatility_range / (np.mean(recent_volatility) + 1e-8)
        
        # Composite Alpha Construction
        core_signal = (opening_acceleration * closing_momentum * daily_compression_ratio) * (concentration_shift * volume_price_divergence)
        momentum_enhanced = core_signal * (position_bias * trend_consistency)
        rejection_adjusted = momentum_enhanced * net_rejection
        final_alpha = rejection_adjusted * volatility_adjustment
        
        result.iloc[i] = final_alpha
    
    return result
