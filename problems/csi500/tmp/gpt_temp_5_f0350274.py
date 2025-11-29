import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Composite Range-Momentum Divergence Factor
    Combines range compression analysis, momentum divergence signals, and volume acceleration
    to generate alpha signals
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required minimum data points
    min_periods = 20
    
    for i in range(min_periods, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # 1. Multi-Timeframe Range Dynamics
        # Calculate 5-day and 10-day high-low ranges
        range_5d = current_data['high'].rolling(window=5, min_periods=5).max() - current_data['low'].rolling(window=5, min_periods=5).min()
        range_10d = current_data['high'].rolling(window=10, min_periods=10).max() - current_data['low'].rolling(window=10, min_periods=10).min()
        
        # Range compression intensity
        if i >= 10:
            compression_intensity = range_5d.iloc[i] / range_10d.iloc[i]
            # Symmetrical vs directional compression (ratio of current range to historical median)
            hist_range_median = range_10d.iloc[i-10:i].median()
            range_ratio = range_5d.iloc[i] / hist_range_median if hist_range_median > 0 else 1.0
        else:
            compression_intensity = 1.0
            range_ratio = 1.0
        
        # Opening range efficiency
        if i >= 1:
            opening_gap = (current_data['open'].iloc[i] / current_data['close'].iloc[i-1]) - 1
            intraday_range = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i])
            total_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            range_utilization = intraday_range / total_range if total_range > 0 else 0
            
            # Gap-range alignment
            gap_direction = 1 if opening_gap > 0 else -1
            intraday_direction = 1 if (current_data['close'].iloc[i] - current_data['open'].iloc[i]) > 0 else -1
            gap_alignment = 1 if gap_direction == intraday_direction else -1
            gap_efficiency = range_utilization * gap_alignment
        else:
            opening_gap = 0
            range_utilization = 0
            gap_efficiency = 0
        
        # 2. Momentum Divergence Signals
        # Multi-scale momentum components
        if i >= 1:
            ultra_short_momentum = (current_data['close'].iloc[i] / current_data['open'].iloc[i]) - 1
            short_term_momentum = (current_data['close'].iloc[i] / current_data['close'].iloc[i-1]) - 1
        else:
            ultra_short_momentum = 0
            short_term_momentum = 0
            
        if i >= 5:
            medium_term_momentum = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5]) - 1
        else:
            medium_term_momentum = 0
        
        # Momentum direction alignment
        ultra_short_dir = 1 if ultra_short_momentum > 0 else -1
        short_term_dir = 1 if short_term_momentum > 0 else -1
        medium_term_dir = 1 if medium_term_momentum > 0 else -1
        
        # Divergence patterns
        us_vs_st_divergence = 1 if ultra_short_dir == short_term_dir else -1
        st_vs_mt_divergence = 1 if short_term_dir == medium_term_dir else -1
        
        # Divergence strength
        divergence_strength = (abs(ultra_short_momentum) + abs(short_term_momentum) + abs(medium_term_momentum)) / 3
        divergence_score = (us_vs_st_divergence + st_vs_mt_divergence) * divergence_strength
        
        # 3. Volume-Enhanced Composite Factor
        # Volume acceleration
        if i >= 5:
            volume_5d_mean = current_data['volume'].iloc[i-5:i].mean()
            current_volume = current_data['volume'].iloc[i]
            volume_momentum = current_volume / volume_5d_mean if volume_5d_mean > 0 else 1.0
            
            # Volume-momentum alignment
            volume_alignment = 1 if (volume_momentum > 1 and short_term_momentum > 0) or (volume_momentum < 1 and short_term_momentum < 0) else -1
            volume_divergence = volume_alignment * abs(volume_momentum - 1)
        else:
            volume_momentum = 1.0
            volume_divergence = 0
        
        # Combine range and divergence signals
        range_divergence_composite = compression_intensity * divergence_score
        
        # Incorporate opening efficiency
        opening_component = gap_efficiency * range_utilization
        
        # Amount confidence multiplier
        if i >= 5:
            amount_5d_mean = current_data['amount'].iloc[i-5:i].mean()
            current_amount = current_data['amount'].iloc[i]
            amount_confidence = current_amount / amount_5d_mean if amount_5d_mean > 0 else 1.0
        else:
            amount_confidence = 1.0
        
        # Intraday price action assessment
        if i >= 1:
            high_low_mid = (current_data['high'].iloc[i] + current_data['low'].iloc[i]) / 2
            close_position = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 0 else 0.5
            breakout_potential = abs(close_position - 0.5) * 2  # Higher when close to extremes
        else:
            breakout_potential = 0
        
        # Generate final composite factor
        alpha_score = (
            range_divergence_composite * 0.4 +
            volume_divergence * 0.3 +
            opening_component * 0.2 +
            breakout_potential * 0.1
        ) * amount_confidence
        
        result.iloc[i] = alpha_score
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
