import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous close for gap calculations
    data['prev_close'] = data['close'].shift(1)
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(2, len(data)):
        current_data = data.iloc[i]
        prev_data = data.iloc[i-1]
        
        # 1. Intraday Price Momentum Asymmetry Analysis
        # Morning Momentum Asymmetry
        morning_asymmetry = ((current_data['high'] - current_data['open']) / current_data['open'] - 
                            (current_data['open'] - current_data['low']) / current_data['low'])
        
        # Afternoon Momentum Asymmetry  
        afternoon_asymmetry = ((current_data['close'] - current_data['low']) / current_data['low'] - 
                              (current_data['high'] - current_data['close']) / current_data['high'])
        
        full_session_asymmetry = morning_asymmetry - afternoon_asymmetry
        
        # Volume timing asymmetry (assuming equal split for demonstration)
        morning_volume_ratio = 0.5  # Placeholder for actual morning volume
        afternoon_volume_ratio = 0.5  # Placeholder for actual afternoon volume
        volume_timing_asymmetry = morning_volume_ratio - afternoon_volume_ratio
        
        # Price-Volume Asymmetry Interaction
        momentum_volume_alignment = morning_asymmetry * morning_volume_ratio
        divergence_detection = full_session_asymmetry - volume_timing_asymmetry
        
        # Asymmetry persistence (3-day average)
        if i >= 4:
            asymmetry_persistence = np.mean([full_session_asymmetry, 
                                           data.iloc[i-1]['close'] - data.iloc[i-1]['open'],
                                           data.iloc[i-2]['close'] - data.iloc[i-2]['open']])
        else:
            asymmetry_persistence = full_session_asymmetry
        
        # 2. Range Efficiency Momentum with Volume Confirmation
        price_range = current_data['high'] - current_data['low']
        if price_range > 0:
            upper_efficiency = (current_data['high'] - current_data['open']) / price_range
            lower_efficiency = (current_data['close'] - current_data['low']) / price_range
            range_efficiency_divergence = upper_efficiency - lower_efficiency
            
            volume_per_range = current_data['volume'] / price_range
            amount_efficiency = current_data['amount'] / current_data['volume'] if current_data['volume'] > 0 else 0
            
            volume_range_alignment = volume_per_range * range_efficiency_divergence
            efficiency_volume_confirmation = range_efficiency_divergence * volume_range_alignment
            
            # Persistence analysis (3-day consistency)
            if i >= 4:
                persistence_analysis = np.std([range_efficiency_divergence,
                                             (data.iloc[i-1]['high'] - data.iloc[i-1]['open']) / (data.iloc[i-1]['high'] - data.iloc[i-1]['low']),
                                             (data.iloc[i-2]['high'] - data.iloc[i-2]['open']) / (data.iloc[i-2]['high'] - data.iloc[i-2]['low'])])
            else:
                persistence_analysis = 1.0
        else:
            range_efficiency_divergence = 0
            efficiency_volume_confirmation = 0
            persistence_analysis = 1.0
            amount_efficiency = 0
        
        # 3. Opening Gap Persistence with Intraday Confirmation
        if not pd.isna(current_data['prev_close']) and current_data['prev_close'] > 0:
            gap_momentum = (current_data['open'] - current_data['prev_close']) / current_data['prev_close']
            gap_persistence = gap_momentum * (current_data['close'] - current_data['open']) / current_data['open']
            gap_direction_consistency = np.sign(gap_momentum) * np.sign(current_data['close'] - current_data['open'])
            
            # Volume-based gap validation
            if prev_data['volume'] > 0:
                gap_volume_alignment = gap_momentum * (current_data['volume'] / prev_data['volume'])
            else:
                gap_volume_alignment = 0
                
            if prev_data['amount'] > 0:
                amount_confirmation = gap_direction_consistency * (current_data['amount'] / prev_data['amount'])
            else:
                amount_confirmation = 0
                
            composite_gap_factor = gap_persistence * gap_volume_alignment * amount_confirmation
        else:
            composite_gap_factor = 0
        
        # 4. Session Transition Momentum with Volume Timing
        # Assuming morning session = first half, afternoon session = second half
        morning_high = current_data['open'] * 1.02  # Placeholder
        morning_low = current_data['open'] * 0.98   # Placeholder
        afternoon_high = current_data['high']
        afternoon_low = current_data['low']
        
        morning_to_afternoon_transition = afternoon_low / morning_high if morning_high > 0 else 1
        session_momentum_transfer = (afternoon_high - morning_low) / morning_low if morning_low > 0 else 0
        
        transition_range = afternoon_high - morning_low
        if transition_range > 0:
            transition_efficiency = (current_data['close'] - morning_low) / transition_range
        else:
            transition_efficiency = 0
        
        # Volume timing patterns (placeholders for actual session volumes)
        morning_volume = current_data['volume'] * 0.5
        afternoon_volume = current_data['volume'] * 0.5
        morning_amount = current_data['amount'] * 0.5
        afternoon_amount = current_data['amount'] * 0.5
        
        if morning_volume > 0:
            volume_acceleration = (afternoon_volume - morning_volume) / morning_volume
        else:
            volume_acceleration = 0
            
        if morning_amount > 0:
            amount_timing = (afternoon_amount - morning_amount) / morning_amount
        else:
            amount_timing = 0
            
        volume_amount_timing_divergence = volume_acceleration - amount_timing
        
        # Transition momentum factor
        momentum_volume_alignment_trans = session_momentum_transfer * volume_acceleration
        efficiency_confirmation = transition_efficiency * volume_amount_timing_divergence
        composite_transition_factor = momentum_volume_alignment_trans * efficiency_confirmation
        
        # Combine all components into final factor
        factor_value = (divergence_detection * persistence_analysis + 
                       efficiency_volume_confirmation * amount_efficiency + 
                       composite_gap_factor + 
                       composite_transition_factor)
        
        factor_values.iloc[i] = factor_value
    
    # Fill NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
