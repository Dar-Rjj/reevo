import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Momentum Dynamics
    # Opening Gap Intensity
    data['gap_intensity'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap Absorption Efficiency
    gap = data['open'] - data['close'].shift(1)
    data['gap_absorption'] = np.where(gap != 0, (data['close'] - data['open']) / gap, 0)
    
    # Gap Decay Analysis
    daily_range_5 = (data['close'].shift(5) - data['open'].shift(5))
    daily_range_10 = (data['close'].shift(10) - data['open'].shift(10))
    daily_range_diff = daily_range_5 - daily_range_10
    current_range_diff = (data['close'] - data['open']) - (data['close'].shift(1) - data['open'].shift(1))
    data['gap_decay'] = np.where(daily_range_diff != 0, current_range_diff / daily_range_diff, 0)
    
    # Intraday Compression Efficiency (simplified using daily OHLC)
    # Morning Compression (first half of day proxy)
    morning_high = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    morning_low = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['morning_compression'] = np.where(
        (morning_high - morning_low) != 0,
        (morning_high - data['open']) / (morning_high - morning_low),
        0
    )
    
    # Afternoon Momentum (second half of day proxy)
    afternoon_high = data['high']
    afternoon_low = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[-1] if len(x) == 2 else np.nan)
    data['afternoon_momentum'] = np.where(
        (afternoon_high - afternoon_low) != 0,
        (data['close'] - afternoon_low) / (afternoon_high - afternoon_low),
        0
    )
    
    # Session Compression Divergence
    data['compression_divergence'] = data['morning_compression'] - data['afternoon_momentum']
    
    # Volume Efficiency Patterns
    # Volume Acceleration
    vol_diff_5_10 = data['volume'].shift(5) - data['volume'].shift(10)
    data['volume_acceleration'] = np.where(
        vol_diff_5_10 != 0,
        (data['volume'] - data['volume'].shift(1)) / vol_diff_5_10,
        0
    )
    
    # Morning Volume Efficiency
    data['morning_volume_efficiency'] = np.where(
        (morning_high - morning_low) != 0,
        data['volume'] * 0.4 / (morning_high - morning_low),  # Assume 40% of volume in morning
        0
    )
    
    # Afternoon Volume Distribution
    data['afternoon_volume_dist'] = data['volume'] * 0.6 / data['volume']  # Assume 60% of volume in afternoon
    
    # Multi-Timeframe Integration
    # Short-term Gap Decay with intraday compression
    data['short_term_gap_decay'] = data['gap_decay'] * data['compression_divergence']
    
    # Medium-term Volume Divergence with gap absorption
    data['medium_term_volume_div'] = data['volume_acceleration'] * data['gap_absorption']
    
    # Long-term Compression Persistence
    data['long_term_compression'] = data['morning_compression'].rolling(window=10, min_periods=5).mean()
    
    # Momentum Quality Assessment
    # Gap-Momentum Direction Conflict
    data['direction_conflict'] = (
        np.sign(data['gap_intensity']) != np.sign(data['afternoon_momentum'])
    ).astype(float)
    
    # Compression Absorption Coherence
    data['compression_coherence'] = data['morning_compression'] * np.abs(data['gap_intensity'])
    
    # Adaptive Alpha Signal Generation
    # Combine gap momentum with compression efficiency
    momentum_compression = data['gap_intensity'] * data['compression_divergence']
    
    # Weight by volume efficiency patterns
    volume_weighted = momentum_compression * data['morning_volume_efficiency']
    
    # Incorporate multi-timeframe gap decay
    multi_timeframe = (
        volume_weighted * 0.4 + 
        data['short_term_gap_decay'] * 0.3 + 
        data['medium_term_volume_div'] * 0.2 + 
        data['long_term_compression'] * 0.1
    )
    
    # Final factor with momentum quality adjustment
    factor = multi_timeframe * (1 - data['direction_conflict']) * data['compression_coherence']
    
    # Clean and return
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
