import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all intermediate columns
    data['morning_fracture'] = 0.0
    data['afternoon_fracture'] = 0.0
    data['session_fracture_divergence'] = 0.0
    data['morning_volume_fracture'] = 0.0
    data['afternoon_volume_fracture'] = 0.0
    data['volume_fracture_timing'] = 0.0
    data['compression_ratio'] = 0.0
    data['fracture_enhanced_compression'] = 0.0
    data['compression_persistence'] = 0.0
    data['fracture_magnitude'] = 0.0
    data['compression_breakout'] = 0.0
    data['volume_confirmed_compression'] = 0.0
    data['gap_absorption'] = 0.0
    data['session_aligned_gap_fill'] = 0.0
    data['compression_scaled_absorption'] = 0.0
    data['price_volume_gap_divergence'] = 0.0
    data['session_timing_gap_alignment'] = 0.0
    data['multi_dimensional_gap_convergence'] = 0.0
    data['morning_utilization'] = 0.0
    data['afternoon_efficiency'] = 0.0
    data['range_divergence'] = 0.0
    data['volume_weighted_range_utilization'] = 0.0
    data['compression_scaled_range_signals'] = 0.0
    data['gap_aligned_range_fracture'] = 0.0
    data['fracture_momentum'] = 0.0
    data['compression_scaled_fracture_persistence'] = 0.0
    data['volume_confirmed_momentum'] = 0.0
    data['gap_enhanced_fracture_momentum'] = 0.0
    data['range_utilized_momentum'] = 0.0
    
    # Calculate basic components with safe division
    for i in range(1, len(data)):
        # Session-Aligned Fracture Detection
        if i >= 1:
            # Simplified morning fracture (using daily data as proxy)
            high_low_range = data['high'].iloc[i] - data['low'].iloc[i]
            if high_low_range > 0:
                data.loc[data.index[i], 'morning_fracture'] = (data['close'].iloc[i] - data['open'].iloc[i]) / high_low_range
                data.loc[data.index[i], 'afternoon_fracture'] = (data['close'].iloc[i] - data['low'].iloc[i]) / high_low_range
            
            data.loc[data.index[i], 'session_fracture_divergence'] = (
                data['morning_fracture'].iloc[i] - data['afternoon_fracture'].iloc[i]
            )
        
        # Volume-Concentrated Fracture Patterns (simplified)
        if i >= 1:
            total_volume = data['volume'].iloc[i]
            if total_volume > 0:
                # Simplified volume fracture patterns
                data.loc[data.index[i], 'morning_volume_fracture'] = (
                    data['morning_fracture'].iloc[i] * 0.5  # Proxy for morning volume ratio
                )
                data.loc[data.index[i], 'afternoon_volume_fracture'] = (
                    data['afternoon_fracture'].iloc[i] * 0.5  # Proxy for afternoon volume ratio
                )
            
            data.loc[data.index[i], 'volume_fracture_timing'] = (
                data['morning_volume_fracture'].iloc[i] - data['afternoon_volume_fracture'].iloc[i]
            )
        
        # Range Compression with Fracture Dynamics
        if i >= 2:
            current_range = data['high'].iloc[i] - data['low'].iloc[i]
            prev_range = data['high'].iloc[i-1] - data['low'].iloc[i-1]
            
            if prev_range > 0 and current_range > 0:
                compression_ratio = current_range / prev_range
                data.loc[data.index[i], 'compression_ratio'] = compression_ratio
                
                # Fracture-enhanced compression
                data.loc[data.index[i], 'fracture_enhanced_compression'] = (
                    compression_ratio * data['morning_fracture'].iloc[i]
                )
                
                # Multi-session compression persistence
                if i >= 3:
                    prev_compression = data['compression_ratio'].iloc[i-1]
                    if prev_compression > 0:
                        data.loc[data.index[i], 'compression_persistence'] = (
                            compression_ratio / prev_compression
                        )
        
        # Compression-Fracture Extremes Detection
        if i >= 1:
            daily_range = data['high'].iloc[i] - data['low'].iloc[i]
            if daily_range > 0:
                fracture_magnitude = abs(data['close'].iloc[i] - data['open'].iloc[i]) / daily_range
                data.loc[data.index[i], 'fracture_magnitude'] = fracture_magnitude
                
                # Compression-breakout potential
                data.loc[data.index[i], 'compression_breakout'] = (
                    fracture_magnitude * data['session_fracture_divergence'].iloc[i]
                )
                
                # Volume-confirmed compression signals
                data.loc[data.index[i], 'volume_confirmed_compression'] = (
                    data['compression_breakout'].iloc[i] * data['volume_fracture_timing'].iloc[i]
                )
        
        # Fracture-Enhanced Gap Absorption
        if i >= 2:
            gap = abs(data['open'].iloc[i] - data['close'].iloc[i-1])
            if gap > 0:
                gap_absorption = (data['close'].iloc[i] - min(data['open'].iloc[i], data['close'].iloc[i-1])) / gap
                data.loc[data.index[i], 'gap_absorption'] = gap_absorption
                
                # Session-aligned gap fill
                data.loc[data.index[i], 'session_aligned_gap_fill'] = (
                    gap_absorption * data['session_fracture_divergence'].iloc[i]
                )
                
                # Compression-scaled absorption
                if data['compression_ratio'].iloc[i] > 0:
                    data.loc[data.index[i], 'compression_scaled_absorption'] = (
                        gap_absorption / data['compression_ratio'].iloc[i]
                    )
        
        # Microstructure Divergence in Gap Dynamics
        if i >= 2:
            if data['gap_absorption'].iloc[i] != 0:
                # Price-volume gap divergence
                volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1.0
                data.loc[data.index[i], 'price_volume_gap_divergence'] = (
                    data['gap_absorption'].iloc[i] / volume_ratio
                )
                
                # Session timing gap alignment
                data.loc[data.index[i], 'session_timing_gap_alignment'] = (
                    data['gap_absorption'].iloc[i] * data['volume_fracture_timing'].iloc[i]
                )
                
                # Multi-dimensional gap convergence
                data.loc[data.index[i], 'multi_dimensional_gap_convergence'] = (
                    data['gap_absorption'].iloc[i] * data['compression_ratio'].iloc[i]
                )
        
        # Fracture-Range Efficiency Dynamics
        if i >= 1:
            daily_range = data['high'].iloc[i] - data['low'].iloc[i]
            if daily_range > 0:
                # Morning range utilization (simplified)
                data.loc[data.index[i], 'morning_utilization'] = (
                    data['morning_fracture'].iloc[i] * 0.5  # Proxy for morning range ratio
                )
                
                # Afternoon fracture efficiency
                data.loc[data.index[i], 'afternoon_efficiency'] = (
                    data['afternoon_fracture'].iloc[i] * 0.5  # Proxy for afternoon range ratio
                )
                
                # Session range divergence
                data.loc[data.index[i], 'range_divergence'] = (
                    data['morning_utilization'].iloc[i] - data['afternoon_efficiency'].iloc[i]
                )
        
        # Volume-Enhanced Range Fracture
        if i >= 2:
            # Volume-weighted range utilization
            volume_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1.0
            data.loc[data.index[i], 'volume_weighted_range_utilization'] = (
                data['range_divergence'].iloc[i] * volume_ratio
            )
            
            # Compression-scaled range signals
            if data['compression_ratio'].iloc[i] > 0:
                data.loc[data.index[i], 'compression_scaled_range_signals'] = (
                    data['range_divergence'].iloc[i] / data['compression_ratio'].iloc[i]
                )
            
            # Gap-aligned range fracture
            data.loc[data.index[i], 'gap_aligned_range_fracture'] = (
                data['range_divergence'].iloc[i] * data['gap_absorption'].iloc[i]
            )
        
        # Multi-Timeframe Fracture Momentum
        if i >= 6:
            # 5-day fracture momentum
            data.loc[data.index[i], 'fracture_momentum'] = (
                data['session_fracture_divergence'].iloc[i] - data['session_fracture_divergence'].iloc[i-5]
            )
            
            # Compression-scaled fracture persistence
            if data['compression_ratio'].iloc[i] > 0:
                data.loc[data.index[i], 'compression_scaled_fracture_persistence'] = (
                    data['fracture_momentum'].iloc[i] / data['compression_ratio'].iloc[i]
                )
            
            # Volume-confirmed momentum
            data.loc[data.index[i], 'volume_confirmed_momentum'] = (
                data['fracture_momentum'].iloc[i] * data['volume_fracture_timing'].iloc[i]
            )
            
            # Gap-enhanced fracture momentum
            data.loc[data.index[i], 'gap_enhanced_fracture_momentum'] = (
                data['fracture_momentum'].iloc[i] * data['gap_absorption'].iloc[i]
            )
            
            # Range-utilized momentum
            data.loc[data.index[i], 'range_utilized_momentum'] = (
                data['fracture_momentum'].iloc[i] * data['range_divergence'].iloc[i]
            )
    
    # Generate Composite Session Microstructure Alpha
    alpha_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 6:  # Ensure we have enough history
            # Primary Fracture-Compression Component
            primary_component = (
                data['session_fracture_divergence'].iloc[i] * data['compression_ratio'].iloc[i] +
                data['volume_fracture_timing'].iloc[i] * data['gap_absorption'].iloc[i] +
                data['range_divergence'].iloc[i] * data['fracture_momentum'].iloc[i]
            )
            
            # Dynamic Microstructure Confirmation
            dynamic_confirmation = (
                data['volume_confirmed_compression'].iloc[i] +
                data['compression_persistence'].iloc[i] +
                data['session_timing_gap_alignment'].iloc[i] +
                data['gap_aligned_range_fracture'].iloc[i]
            )
            
            # Momentum Integration System
            momentum_integration = (
                data['session_fracture_divergence'].iloc[i] +
                data['compression_scaled_absorption'].iloc[i] +
                data['volume_weighted_range_utilization'].iloc[i] +
                data['multi_dimensional_gap_convergence'].iloc[i]
            )
            
            # Composite alpha factor
            alpha_values.iloc[i] = (
                0.4 * primary_component +
                0.3 * dynamic_confirmation +
                0.3 * momentum_integration
            )
        else:
            alpha_values.iloc[i] = 0.0
    
    return alpha_values
