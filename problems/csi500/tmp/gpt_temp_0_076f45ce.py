import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize all required columns with NaN
    for col in ['prev_close', 'prev_high', 'prev_low', 'prev_volume', 'prev_amount', 
                'prev_prev_high', 'prev_prev_low', 'prev_range']:
        data[col] = np.nan
    
    # Calculate previous values using shift(1) for one-day lag
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    data['prev_prev_high'] = data['high'].shift(2)
    data['prev_prev_low'] = data['low'].shift(2)
    data['prev_range'] = (data['prev_high'] - data['prev_low']).replace(0, np.nan)
    
    # Fractal Volatility Cascade with Volume Dynamics
    volatility_cascade = ((data['high'] - data['low']) / (data['prev_high'] - data['prev_low']) * 
                         (data['high'] - data['low']) / (data['prev_prev_high'] - data['prev_prev_low']))
    volume_enhanced_cascade = volatility_cascade * (data['volume'] / data['prev_volume'])
    fractal_cascade_divergence = volatility_cascade - volume_enhanced_cascade
    
    # Volume-Weighted Volatility Confirmation
    first_half_intensity = ((data['high'] - data['open']) / (data['high'] - data['low'])) * data['volume']
    second_half_intensity = ((data['close'] - data['low']) / (data['high'] - data['low'])) * data['volume']
    volatility_fractal_imbalance = first_half_intensity - second_half_intensity
    
    # Opening-Closing Momentum with Gap Dynamics
    opening_gap_pressure = (data['open'] - data['prev_close']) * (data['amount'] / data['prev_amount'])
    closing_momentum_pressure = (data['close'] - data['open']) * (data['volume'] / data['prev_volume'])
    gap_momentum_asymmetry = opening_gap_pressure - closing_momentum_pressure
    
    # Range-Adjusted Gap Validation
    gap_absorption_efficiency = ((data['high'] - data['open']) / (data['open'] - data['prev_close'])) * data['volume']
    closing_range_capture = ((data['close'] - data['low']) / (data['high'] - data['low'])) * data['amount']
    gap_range_efficiency = gap_absorption_efficiency * closing_range_capture
    
    # Regime-Adaptive Volatility Momentum
    upward_vol_signal = ((data['high'] - data['open']) > (data['close'] - data['low'])) * data['volume']
    downward_vol_signal = ((data['high'] - data['open']) < (data['close'] - data['low'])) * data['volume']
    net_volatility_flow = upward_vol_signal - downward_vol_signal
    
    # Volume-Enhanced Volatility Persistence
    volatility_volume_confirmation = net_volatility_flow * (data['volume'] / data['prev_volume'])
    
    # Calculate regime duration
    regime_duration = pd.Series(index=data.index, dtype=float)
    current_duration = 0
    prev_regime = 0
    
    for i in range(len(data)):
        if i < 2:
            regime_duration.iloc[i] = 0
            continue
            
        current_regime = 1 if net_volatility_flow.iloc[i] > 0 else (-1 if net_volatility_flow.iloc[i] < 0 else 0)
        
        if current_regime == prev_regime and prev_regime != 0:
            current_duration += 1
        else:
            current_duration = 1 if current_regime != 0 else 0
        
        regime_duration.iloc[i] = current_duration
        prev_regime = current_regime
    
    volatility_regime_duration = regime_duration * net_volatility_flow
    volatility_momentum = volatility_regime_duration * volatility_volume_confirmation
    
    # Price Range Evolution with Volatility Dynamics
    current_range_efficiency = ((data['high'] - data['low']) / data['prev_range']) * data['volume']
    range_boundary_behavior = ((data['close'] - data['prev_close']) / (data['high'] - data['low'])) * data['amount']
    range_expansion_momentum = current_range_efficiency * range_boundary_behavior
    
    # Micro-Structure Volatility Patterns
    compression_expansion_signal = ((data['high'] - data['low']) / data['prev_range']) * (data['volume'] / data['prev_volume']) * (data['close'] / data['prev_close'])
    
    # Calculate volatility persistence
    vol_persistence = pd.Series(index=data.index, dtype=float)
    current_persistence = 0
    
    for i in range(len(data)):
        if i < 2:
            vol_persistence.iloc[i] = 0
            continue
            
        if (data['high'].iloc[i] - data['low'].iloc[i]) > (data['prev_high'].iloc[i] - data['prev_low'].iloc[i]):
            current_persistence += 1
        else:
            current_persistence = 0
        
        vol_persistence.iloc[i] = current_persistence
    
    volatility_persistence = vol_persistence * range_expansion_momentum
    volatility_range_coherence = compression_expansion_signal * volatility_persistence
    
    # Volume-Price Volatility Alignment (simplified without intraday data)
    first_half_alignment = ((data['high'] - data['open']) / (data['high'] - data['low'])) * 0.5
    second_half_alignment = ((data['close'] - data['low']) / (data['high'] - data['low'])) * 0.5
    volume_volatility_divergence = first_half_alignment - second_half_alignment
    
    # Temporal Volume Concentration (simplified without intraday data)
    early_session_bias = 0.3 * (data['open'] - data['prev_close']) * data['volume']
    late_session_bias = 0.3 * (data['close'] - data['open']) * data['volume']
    volume_temporal_imbalance = early_session_bias - late_session_bias
    
    # Momentum-Volatility Coherence
    overnight_momentum = ((data['open'] - data['prev_close']) / data['prev_close']) * data['volume']
    intraday_momentum = ((data['close'] - data['open']) / data['open']) * data['amount']
    momentum_volatility_coherence = overnight_momentum * intraday_momentum * data['volume']
    
    # Volume-Concentrated Volatility Signals
    volume_accumulation_efficiency = data['volume'] * ((data['close'] - data['open']) / (data['high'] - data['low']))
    
    # Calculate volume drought persistence
    volume_drought = pd.Series(index=data.index, dtype=float)
    current_drought = 0
    
    for i in range(len(data)):
        if i < 1:
            volume_drought.iloc[i] = 0
            continue
            
        if data['volume'].iloc[i] < data['prev_volume'].iloc[i]:
            current_drought += 1
        else:
            current_drought = 0
        
        volume_drought.iloc[i] = current_drought
    
    volume_drought_persistence = -volume_drought * (data['close'] - data['open'])
    volume_volatility_alignment = volume_accumulation_efficiency * volume_drought_persistence
    
    # Behavioral Volatility Patterns
    herding_signal = data['volume'] * ((data['close'] - data['open']) / (data['high'] - data['low']))
    anti_herding_opportunity = -data['volume'] * ((data['high'] - data['low']) / data['prev_range'])
    behavioral_divergence = herding_signal * anti_herding_opportunity
    
    # Micro-Move Coalescence
    fragmented_trend_assembly = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) * data['volume']
    micro_breakout_confirmation = ((data['high'] - data['low'] - data['prev_range']) / data['prev_range']) * data['amount']
    behavioral_efficiency = fragmented_trend_assembly * micro_breakout_confirmation
    
    # Composite Volatility Alpha Synthesis
    volatility_cascade_signal = fractal_cascade_divergence * volatility_fractal_imbalance
    gap_dynamics_signal = gap_momentum_asymmetry * gap_range_efficiency
    volatility_momentum_signal = volatility_momentum * net_volatility_flow
    range_volatility_signal = volatility_range_coherence * range_expansion_momentum
    volume_alignment_signal = volume_volatility_divergence * volume_temporal_imbalance
    momentum_coherence_signal = momentum_volatility_coherence * volume_volatility_alignment
    behavioral_pattern_signal = behavioral_divergence * behavioral_efficiency
    
    # Final Alpha Factor
    final_alpha = (volatility_cascade_signal * gap_dynamics_signal * volatility_momentum_signal * 
                  range_volatility_signal * volume_alignment_signal * momentum_coherence_signal * 
                  behavioral_pattern_signal)
    
    return final_alpha
