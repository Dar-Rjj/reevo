import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling windows for multi-timeframe analysis
    for i in range(len(df)):
        if i < 5:  # Need at least 5 days of history
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1] if i >= 1 else None
        prev2_data = df.iloc[i-2] if i >= 2 else None
        prev5_data = df.iloc[i-5] if i >= 5 else None
        
        # Multi-Timeframe Range Compression
        short_term_range = current_data['high'] - current_data['low']
        long_term_range = prev5_data['high'] - prev5_data['low'] if prev5_data is not None else 1
        compression_ratio = short_term_range / long_term_range if long_term_range != 0 else 1
        
        prev_compression_ratio = (prev_data['high'] - prev_data['low']) / (df.iloc[i-6]['high'] - df.iloc[i-6]['low']) if i >= 6 and (df.iloc[i-6]['high'] - df.iloc[i-6]['low']) != 0 else 1
        compression_momentum = compression_ratio / prev_compression_ratio if prev_compression_ratio != 0 else 1
        dynamic_compression_factor = compression_ratio * compression_momentum
        
        # Session-Based Volatility Efficiency (using current day data as proxy)
        morning_vol_capture = (current_data['high'] - current_data['open']) / (current_data['high'] - current_data['low']) if (current_data['high'] - current_data['low']) != 0 else 0
        afternoon_consolidation = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low']) if (current_data['high'] - current_data['low']) != 0 else 0
        session_efficiency_ratio = morning_vol_capture / afternoon_consolidation if afternoon_consolidation != 0 else 1
        
        # Volume Clustering Dynamics (using current day volume as total proxy)
        early_session_concentration = 0.4  # Conservative estimate for first 30min
        late_session_fragmentation = 0.3   # Conservative estimate for last 30min
        volume_timing_asymmetry = early_session_concentration / late_session_fragmentation if late_session_fragmentation != 0 else 1
        
        # Price-Volume Microstructure
        volume_weighted_range_eff = ((current_data['close'] - current_data['open']) * current_data['volume']) / (current_data['high'] - current_data['low']) if (current_data['high'] - current_data['low']) != 0 else 0
        range_volume_density = current_data['volume'] / (current_data['high'] - current_data['low']) if (current_data['high'] - current_data['low']) != 0 else 0
        microstructure_synchronization = volume_weighted_range_eff * range_volume_density
        
        # Opening Auction Dynamics
        opening_momentum_capture = (current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low']) if (current_data['high'] - current_data['low']) != 0 else 0
        opening_range_utilization = (current_data['high'] - current_data['open']) / (current_data['open'] - current_data['low']) if (current_data['open'] - current_data['low']) != 0 else 1
        opening_efficiency_score = opening_momentum_capture * opening_range_utilization
        
        # Closing Auction Efficiency
        closing_pressure_intensity = (current_data['close'] - current_data['open']) * current_data['volume']
        closing_range_completion = (current_data['close'] - current_data['low']) / (current_data['high'] - current_data['low']) if (current_data['high'] - current_data['low']) != 0 else 0
        closing_momentum_persistence = closing_pressure_intensity * closing_range_completion
        
        # Volatility Memory Effects
        previous_close_influence = (current_data['open'] - prev_data['close']) / (current_data['high'] - current_data['low']) if (current_data['high'] - current_data['low']) != 0 else 0
        intraday_reversal_tendency = (current_data['close'] - current_data['open']) / (current_data['open'] - prev_data['close']) if (current_data['open'] - prev_data['close']) != 0 else 1
        volatility_memory_factor = previous_close_influence * intraday_reversal_tendency
        
        # Volume Flow Persistence
        if i >= 2:
            vol_momentum1 = current_data['volume'] / prev_data['volume'] if prev_data['volume'] != 0 else 1
            vol_momentum2 = prev_data['volume'] / prev2_data['volume'] if prev2_data['volume'] != 0 else 1
            consecutive_volume_momentum = vol_momentum1 * vol_momentum2
            volume_acceleration = vol_momentum1 / vol_momentum2 if vol_momentum2 != 0 else 1
            flow_persistence_score = consecutive_volume_momentum * volume_acceleration
        else:
            flow_persistence_score = 1
        
        # Composite factors
        volatility_compression_factor = dynamic_compression_factor * session_efficiency_ratio
        volume_synchronization_factor = microstructure_synchronization * volume_timing_asymmetry
        auction_efficiency_factor = opening_efficiency_score * closing_momentum_persistence
        
        # Integrated signals
        short_term_vol_efficiency = volatility_compression_factor * volume_synchronization_factor
        medium_term_microstructure = auction_efficiency_factor * flow_persistence_score
        
        # Final alpha calculation
        final_alpha = short_term_vol_efficiency / (1 + abs(volatility_memory_factor))
        
        result.iloc[i] = final_alpha
    
    return result
