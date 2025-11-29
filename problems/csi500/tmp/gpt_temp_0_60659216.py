import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for current_date in df.index:
        current_idx = df.index.get_loc(current_date)
        
        # Skip if we don't have enough historical data
        if current_idx < 6:
            result.loc[current_date] = np.nan
            continue
        
        current_data = df.iloc[current_idx]
        current_open = current_data['open']
        current_high = current_data['high']
        current_low = current_data['low']
        current_close = current_data['close']
        current_volume = current_data['volume']
        
        # Momentum Acceleration Components
        early_momentum = (current_high - current_open) / current_open if current_open != 0 else 0
        late_momentum = (current_close - current_low) / current_low if current_low != 0 else 0
        momentum_divergence = early_momentum - late_momentum
        
        high_low_range = current_high - current_low
        price_acceleration = (current_close - current_open) / high_low_range if high_low_range != 0 else 0
        
        # Volume-Pressure Analysis
        volume_flow = current_volume * (current_close - current_open) / high_low_range if high_low_range != 0 else 0
        high_pressure = (current_high - current_close) * current_volume
        low_pressure = (current_close - current_low) * current_volume
        
        # Net Pressure over past 3 days
        net_pressure = 0
        for i in range(1, 4):
            if current_idx - i >= 0:
                hist_data = df.iloc[current_idx - i]
                hist_high_pressure = (hist_data['high'] - hist_data['close']) * hist_data['volume']
                hist_low_pressure = (hist_data['close'] - hist_data['low']) * hist_data['volume']
                net_pressure += (hist_low_pressure - hist_high_pressure)
        
        # Acceleration Integration
        acceleration_weighted_divergence = momentum_divergence * price_acceleration
        
        # Volume Flow Divergence (current vs 6-day average)
        volume_flow_sum = 0
        volume_flow_count = 0
        for i in range(6):
            if current_idx - i >= 0:
                hist_data = df.iloc[current_idx - i]
                hist_high_low_range = hist_data['high'] - hist_data['low']
                hist_volume_flow = hist_data['volume'] * (hist_data['close'] - hist_data['open']) / hist_high_low_range if hist_high_low_range != 0 else 0
                volume_flow_sum += hist_volume_flow
                volume_flow_count += 1
        
        avg_volume_flow = volume_flow_sum / volume_flow_count if volume_flow_count > 0 else 0
        volume_flow_divergence = volume_flow - avg_volume_flow
        
        flow_enhanced_divergence = acceleration_weighted_divergence * volume_flow_divergence
        
        # Historical Momentum Bias
        volume_weighted_early_momentum = 0
        volume_weighted_late_momentum = 0
        
        for i in range(6):
            if current_idx - i >= 0:
                hist_data = df.iloc[current_idx - i]
                hist_early_momentum = (hist_data['high'] - hist_data['open']) / hist_data['open'] if hist_data['open'] != 0 else 0
                hist_late_momentum = (hist_data['close'] - hist_data['low']) / hist_data['low'] if hist_data['low'] != 0 else 0
                volume_weighted_early_momentum += hist_early_momentum * hist_data['volume']
                volume_weighted_late_momentum += hist_late_momentum * hist_data['volume']
        
        historical_bias = volume_weighted_late_momentum - volume_weighted_early_momentum
        
        # Final Alpha Construction
        base_signal = flow_enhanced_divergence * historical_bias
        pressure_adjustment = base_signal * np.sign(net_pressure) if net_pressure != 0 else base_signal
        
        final_alpha = pressure_adjustment * high_low_range / current_close if current_close != 0 else pressure_adjustment
        
        result.loc[current_date] = final_alpha
    
    return result
