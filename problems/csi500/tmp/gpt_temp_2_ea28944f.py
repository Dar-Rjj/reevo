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
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling windows for complexity measures
    for i in range(len(data)):
        if i < 10:  # Need at least 10 days of data
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use data up to current day
        
        # 1. Price Path Fractal Dimension Analysis
        # Short-term complexity (5-day)
        if i >= 4:
            short_term_numerator = sum(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]) 
                                     for j in range(i-3, i+1))
            short_term_denominator = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-4])
            short_term_complexity = short_term_numerator / short_term_denominator if short_term_denominator != 0 else 0
        else:
            short_term_complexity = 0
            
        # Medium-term complexity (10-day)
        if i >= 9:
            medium_term_numerator = sum(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]) 
                                      for j in range(i-8, i+1))
            medium_term_denominator = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-9])
            medium_term_complexity = medium_term_numerator / medium_term_denominator if medium_term_denominator != 0 else 0
        else:
            medium_term_complexity = 0
            
        path_complexity_divergence = medium_term_complexity - short_term_complexity
        
        # 2. Intraday Price-Volume Co-Movement Patterns
        # For simplicity, using first and last hour approximations
        # Opening efficiency (using first hour high and open)
        opening_efficiency = (current_data['high'].iloc[i] - current_data['open'].iloc[i]) / current_data['volume'].iloc[i] if current_data['volume'].iloc[i] != 0 else 0
        
        # Closing efficiency (using close and last hour low)
        closing_efficiency = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / current_data['volume'].iloc[i] if current_data['volume'].iloc[i] != 0 else 0
        
        efficiency_gap = opening_efficiency - closing_efficiency
        
        # Price-Level Volume Distribution
        daily_midpoint = (current_data['high'].iloc[i] + current_data['low'].iloc[i]) / 2
        # Simplified volume distribution (assuming uniform distribution)
        upper_ratio = 0.5  # Placeholder
        lower_intensity = 0.5  # Placeholder
        volume_distribution_skew = upper_ratio - lower_intensity
        
        # 3. Multi-Scale Reversion-Persistence Dynamics
        # Intraday reversion
        intraday_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        intraday_reversion = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / intraday_range if intraday_range != 0 else 0
        
        # Overnight gap persistence
        if i >= 1:
            prev_day_range = current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]
            overnight_persistence = abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) / prev_day_range if prev_day_range != 0 else 0
        else:
            overnight_persistence = 0
            
        reversion_scale_ratio = intraday_reversion / overnight_persistence if overnight_persistence != 0 else 0
        
        # Volume-weighted path efficiency
        volume_weighted_complexity = path_complexity_divergence * efficiency_gap
        
        # 4. Cross-Sectional Volume Flow Regimes
        # Simplified volume concentration measures
        morning_intensity = current_data['volume'].iloc[i] * (1 if current_data['volume'].iloc[i] > (current_data['volume'].iloc[i-1] if i >= 1 else 0) else 0)
        afternoon_diffusion = current_data['volume'].iloc[i] * (1 if current_data['volume'].iloc[i] < (current_data['volume'].iloc[i-1] if i >= 1 else 0) else 0)
        concentration_diffusion_spread = morning_intensity - afternoon_diffusion
        
        # Intraday range utilization
        intraday_price_move = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i])
        range_utilization = intraday_range / intraday_price_move if intraday_price_move != 0 else 0
        volume_flow_efficiency = range_utilization * current_data['volume'].iloc[i]
        
        # 5. Composite Path Complexity-Volume Alpha
        # Core components
        core_component_1 = volume_weighted_complexity
        core_component_2 = volume_distribution_skew * reversion_scale_ratio
        core_component_3 = concentration_diffusion_spread
        
        # Dynamic signal refinement
        extreme_complexity_detection = 1 if abs(path_complexity_divergence) > 1 else 0
        volume_flow_validation = 1 if volume_flow_efficiency > np.percentile([volume_flow_efficiency], 50) else 0
        
        # Final composite alpha
        composite_alpha = (
            core_component_1 * 0.4 +
            core_component_2 * 0.3 +
            core_component_3 * 0.2 +
            extreme_complexity_detection * 0.05 +
            volume_flow_validation * 0.05
        )
        
        result.iloc[i] = composite_alpha
    
    return result
