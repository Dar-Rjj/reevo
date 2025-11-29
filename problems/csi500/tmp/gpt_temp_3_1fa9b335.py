import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor storage
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(2, len(data)):
        current_date = data.index[i]
        current_data = data.iloc[:i+1]  # Only use current and past data
        
        # 1. Intraday Volatility Compression Breakout
        # Calculate daily range efficiency
        range_efficiency = (current_data['high'] - current_data['low']) / current_data['open']
        
        # Detect compression duration (consecutive decreasing range days)
        compression_duration = 0
        for j in range(min(5, i), 0, -1):
            if range_efficiency.iloc[j] < range_efficiency.iloc[j-1]:
                compression_duration += 1
            else:
                break
        
        # Volume accumulation during compression
        if compression_duration >= 2:
            compression_volume = current_data['volume'].iloc[-compression_duration:].mean()
            normal_volume = current_data['volume'].iloc[-compression_duration-5:-compression_duration].mean()
            volume_accumulation = compression_volume / normal_volume if normal_volume > 0 else 1
        else:
            volume_accumulation = 1
        
        # Price positioning near boundaries
        current_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        if current_range > 0:
            upper_distance = (current_data['high'].iloc[-1] - current_data['close'].iloc[-1]) / current_range
            lower_distance = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / current_range
            boundary_alignment = min(upper_distance, lower_distance)
        else:
            boundary_alignment = 0.5
        
        volatility_factor = compression_duration * volume_accumulation * (1 - boundary_alignment)
        
        # 2. Opening Gap Fade Efficiency
        if i >= 1:
            gap_magnitude = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / current_data['close'].iloc[-2]
            
            # Historical fade success rate (using past 10 gaps)
            fade_success = 0
            gap_count = 0
            for j in range(max(1, i-10), i):
                gap = (current_data['open'].iloc[j] - current_data['close'].iloc[j-1]) / current_data['close'].iloc[j-1]
                if abs(gap) > 0.005:  # Significant gap
                    gap_count += 1
                    if gap > 0 and current_data['close'].iloc[j] < current_data['open'].iloc[j]:  # Gap up, closed down
                        fade_success += 1
                    elif gap < 0 and current_data['close'].iloc[j] > current_data['open'].iloc[j]:  # Gap down, closed up
                        fade_success += 1
            
            fade_probability = fade_success / max(gap_count, 1)
            
            # Volume confirmation
            opening_volume_intensity = current_data['volume'].iloc[-1] / current_data['volume'].iloc[-5:].mean()
            
            gap_factor = abs(gap_magnitude) * fade_probability * opening_volume_intensity * (-1 if gap_magnitude > 0 else 1)
        else:
            gap_factor = 0
        
        # 3. Price-Volume Momentum Divergence
        if i >= 3:
            # 3-day price change rate
            price_change = (current_data['close'].iloc[-1] - current_data['close'].iloc[-4]) / current_data['close'].iloc[-4]
            
            # 3-day volume change rate
            volume_change = (current_data['volume'].iloc[-1] - current_data['volume'].iloc[-4]) / current_data['volume'].iloc[-4]
            
            # Divergence magnitude
            price_trend = np.sign(price_change)
            volume_trend = np.sign(volume_change)
            
            if price_trend != volume_trend:
                divergence_magnitude = abs(price_change - volume_change)
                # Resolution probability based on historical patterns
                resolution_count = 0
                total_divergence = 0
                for j in range(3, min(i, 20)):
                    p_change = (current_data['close'].iloc[j] - current_data['close'].iloc[j-3]) / current_data['close'].iloc[j-3]
                    v_change = (current_data['volume'].iloc[j] - current_data['volume'].iloc[j-3]) / current_data['volume'].iloc[j-3]
                    if np.sign(p_change) != np.sign(v_change):
                        total_divergence += 1
                        # Check if price followed volume direction in next period
                        next_p_change = (current_data['close'].iloc[j+1] - current_data['close'].iloc[j]) / current_data['close'].iloc[j]
                        if np.sign(next_p_change) == np.sign(v_change):
                            resolution_count += 1
                
                resolution_probability = resolution_count / max(total_divergence, 1)
                divergence_factor = divergence_magnitude * resolution_probability * volume_trend
            else:
                divergence_factor = 0
        else:
            divergence_factor = 0
        
        # 4. Amount-Price Efficiency Regime
        if i >= 5:
            # Price change per unit amount (efficiency)
            price_change_5d = (current_data['close'].iloc[-1] - current_data['close'].iloc[-6]) / current_data['close'].iloc[-6]
            total_amount_5d = current_data['amount'].iloc[-5:].sum()
            
            if total_amount_5d > 0:
                efficiency = price_change_5d / (total_amount_5d / 1e6)  # Normalize by million
            else:
                efficiency = 0
            
            # Efficiency trend detection (5-day rolling)
            efficiency_trend = 0
            if i >= 10:
                prev_efficiency = (current_data['close'].iloc[-6] - current_data['close'].iloc[-11]) / current_data['close'].iloc[-11]
                prev_total_amount = current_data['amount'].iloc[-10:-5].sum()
                if prev_total_amount > 0:
                    prev_efficiency = prev_efficiency / (prev_total_amount / 1e6)
                    efficiency_trend = efficiency - prev_efficiency
                else:
                    efficiency_trend = 0
            
            # Regime classification
            if efficiency_trend > 0:
                regime_strength = 1
            elif efficiency_trend < 0:
                regime_strength = -1
            else:
                regime_strength = 0
            
            efficiency_factor = efficiency * regime_strength
        else:
            efficiency_factor = 0
        
        # Combine factors with weights
        combined_factor = (
            0.3 * volatility_factor +
            0.25 * gap_factor +
            0.25 * divergence_factor +
            0.2 * efficiency_factor
        )
        
        factor_values.loc[current_date] = combined_factor
    
    # Fill initial NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
