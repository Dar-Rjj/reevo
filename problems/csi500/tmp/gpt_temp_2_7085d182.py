import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Create copies to avoid SettingWithCopyWarning
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        current_date = df.index[i]
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1]
        prev_prev_data = df.iloc[i-2]
        
        # Morning vs Afternoon Volatility Structure
        # Assuming first hour: 9:30-10:30, last hour: 15:00-16:00
        # For simplicity, using daily OHLC as proxy for intraday ranges
        morning_volatility = (current_data['high'] - current_data['low']) / current_data['open']
        midday_price = (current_data['high'] + current_data['low']) / 2
        afternoon_volatility = (current_data['high'] - current_data['low']) / midday_price
        
        # Volatility Ratio and Regime Classification
        volatility_ratio = np.log(morning_volatility / afternoon_volatility) if afternoon_volatility > 0 else 0
        
        # Regime thresholds (adjustable parameters)
        compression_threshold = 0.1
        expansion_threshold = -0.1
        
        compression_regime = 1 if volatility_ratio > compression_threshold else 0
        expansion_regime = 1 if volatility_ratio < expansion_threshold else 0
        neutral_regime = 1 if (volatility_ratio >= expansion_threshold and volatility_ratio <= compression_threshold) else 0
        
        # Momentum Acceleration Framework
        high_low_range = current_data['high'] - current_data['low']
        raw_momentum = (current_data['close'] - current_data['open']) / high_low_range if high_low_range > 0 else 0
        
        prev_high_low_range = prev_data['high'] - prev_data['low']
        prev_raw_momentum = (prev_data['close'] - prev_data['open']) / prev_high_low_range if prev_high_low_range > 0 else 0
        
        prev_prev_high_low_range = prev_prev_data['high'] - prev_prev_data['low']
        prev_prev_raw_momentum = (prev_prev_data['close'] - prev_prev_data['open']) / prev_prev_high_low_range if prev_prev_high_low_range > 0 else 0
        
        momentum_persistence = raw_momentum * prev_raw_momentum
        
        current_divergence = raw_momentum - prev_raw_momentum
        previous_divergence = prev_raw_momentum - prev_prev_raw_momentum
        momentum_acceleration = current_divergence - previous_divergence
        
        # Volume-Weighted Intraday Dynamics
        volume_weighted_position = ((current_data['close'] - current_data['low']) / high_low_range) * current_data['volume'] if high_low_range > 0 else 0
        
        # Volume ratios using daily volume as proxy for intraday volumes
        prev_avg_volume = (prev_data['volume'] + prev_prev_data['volume']) / 2 if i >= 2 else prev_data['volume']
        morning_volume_ratio = current_data['volume'] / prev_avg_volume if prev_avg_volume > 0 else 1
        
        # Using current volume as proxy for both morning and afternoon volumes
        afternoon_volume_ratio = current_data['volume'] / current_data['volume'] if current_data['volume'] > 0 else 1
        
        volume_breakout = (current_data['volume'] / prev_data['volume'] - 1) if prev_data['volume'] > 0 else 0
        
        # Regime-Adaptive Signal Integration
        # Compression Regime Signals
        volume_weighted_momentum = volume_weighted_position * raw_momentum
        acceleration_enhancement = momentum_acceleration * volume_breakout
        compression_factor = volume_weighted_momentum * acceleration_enhancement
        
        # Expansion Regime Signals
        reversal_strength = (current_data['close'] - current_data['open']) / high_low_range if high_low_range > 0 else 0
        volume_confirmation = afternoon_volume_ratio * morning_volume_ratio
        expansion_factor = reversal_strength * volume_confirmation
        
        # Neutral Regime Signals
        momentum_persistence_signal = np.sign(current_data['close'] - current_data['open']) * ((current_data['close'] - current_data['low']) / high_low_range) if high_low_range > 0 else 0
        volume_stability = current_data['volume'] / prev_prev_data['volume'] if prev_prev_data['volume'] > 0 else 1
        neutral_factor = momentum_persistence_signal * volume_stability
        
        # Final Alpha Factor Construction
        compression_weight = compression_factor * compression_regime
        expansion_weight = expansion_factor * expansion_regime
        neutral_weight = neutral_factor * neutral_regime
        
        base_acceleration = momentum_acceleration * volume_weighted_position
        regime_modulation = base_acceleration * (compression_weight + expansion_weight + neutral_weight)
        final_factor = regime_modulation * momentum_persistence
        
        result.loc[current_date] = final_factor
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
