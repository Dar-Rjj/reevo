import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum Fragmentation and Reversal Patterns Alpha Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic intraday metrics (assuming 6.5 hour trading day)
    # For simplicity, we'll use approximations since we don't have exact intraday timestamps
    
    # Daily returns and ranges
    daily_return = (df['close'] - df['open']) / df['open']
    daily_range = (df['high'] - df['low']) / df['open']
    
    # Volume metrics
    total_volume = df['volume']
    
    # Calculate fragmentation components
    for i in range(2, len(df)):
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1]
        
        # Skip if insufficient data
        if pd.isna(current_data).any() or pd.isna(prev_data).any():
            continue
            
        # 1. Momentum Fragmentation Score
        # Calculate sub-session momentum variations
        momentum_fragmentation = 0
        
        # Early session momentum (first 30-min approximation)
        early_momentum = (current_data['high'] - current_data['open']) / current_data['open']
        
        # Mid-session momentum (middle of day approximation)
        mid_momentum = (current_data['close'] - (current_data['high'] + current_data['low']) / 2) / current_data['open']
        
        # Late session momentum (last hour approximation)
        late_momentum = (current_data['close'] - current_data['low']) / current_data['open']
        
        # Calculate momentum fragmentation as variance of sub-session returns
        momentum_components = [early_momentum, mid_momentum, late_momentum]
        momentum_fragmentation = np.var(momentum_components) if len(momentum_components) > 1 else 0
        
        # 2. Volume Fragmentation Score
        volume_fragmentation = 0
        
        # Volume concentration metrics
        # Assuming volume is concentrated in certain periods (we'll use price-based proxies)
        volume_variance = 0.1  # Placeholder - in practice would use intraday volume data
        
        # 3. Range Position Variability
        range_fragmentation = 0
        
        # Calculate range position changes throughout the day
        open_range_pos = (current_data['open'] - current_data['low']) / daily_range.iloc[i]
        high_range_pos = (current_data['high'] - current_data['low']) / daily_range.iloc[i]  # Should be 1.0
        close_range_pos = (current_data['close'] - current_data['low']) / daily_range.iloc[i]
        
        range_positions = [open_range_pos, high_range_pos, close_range_pos]
        range_fragmentation = np.var(range_positions) if len(range_positions) > 1 else 0
        
        # 4. Session Boundary Effects
        session_boundary_effect = 0
        
        # Compare current day's opening with previous close momentum
        gap_momentum = (current_data['open'] - prev_data['close']) / prev_data['close']
        opening_momentum = (current_data['high'] - current_data['open']) / current_data['open']
        
        session_boundary_effect = abs(gap_momentum - opening_momentum)
        
        # 5. Volume Confirmation Metrics
        volume_confirmation = 0
        
        # Volume trends throughout the day (approximated)
        volume_trend = 1.0  # Placeholder - would use actual intraday volume patterns
        
        # Composite Fragmentation Score
        fragmentation_composite = (
            momentum_fragmentation * 0.4 +
            volume_fragmentation * 0.3 +
            range_fragmentation * 0.2 +
            session_boundary_effect * 0.1
        )
        
        # Volume-adjusted fragmentation signal
        volume_adjusted_fragmentation = fragmentation_composite * volume_trend
        
        # Generate alpha signal
        # High fragmentation suggests mean reversion, low fragmentation suggests momentum continuation
        if fragmentation_composite > 0.02:  # High fragmentation threshold
            # Mean reversion bias
            alpha_signal = -volume_adjusted_fragmentation * daily_return.iloc[i]
        elif fragmentation_composite < 0.005:  # Low fragmentation threshold
            # Momentum continuation bias
            alpha_signal = volume_adjusted_fragmentation * daily_return.iloc[i]
        else:
            # Neutral/transition regime
            alpha_signal = 0
            
        alpha.iloc[i] = alpha_signal
    
    # Fill NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
