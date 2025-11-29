import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Price Structure
    data['daily_range'] = data['high'] - data['low']
    data['open_position_ratio'] = (data['open'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['close_position_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Track Momentum Divergence
    data['position_change'] = data['close_position_ratio'] - data['open_position_ratio']
    
    # Range momentum
    data['range_change_pct'] = data['daily_range'].pct_change()
    
    # Position change persistence
    data['position_change_sign'] = np.sign(data['position_change'])
    data['position_change_magnitude'] = abs(data['position_change'])
    
    # Track consecutive position changes (same direction)
    data['position_change_consistency'] = 0
    for i in range(1, len(data)):
        if data['position_change_sign'].iloc[i] == data['position_change_sign'].iloc[i-1]:
            data['position_change_consistency'].iloc[i] = data['position_change_consistency'].iloc[i-1] + 1
        else:
            data['position_change_consistency'].iloc[i] = 0
    
    # Volume Confirmation System
    data['volume_change_pct'] = data['volume'].pct_change()
    data['volume_5day_avg'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_spike_ratio'] = data['volume'] / data['volume_5day_avg']
    
    # Volume-Price Alignment
    data['volume_trend_alignment'] = 0
    mask_positive = (data['position_change'] > 0) & (data['volume_change_pct'] > 0)
    mask_negative = (data['position_change'] < 0) & (data['volume_change_pct'] < 0)
    
    data.loc[mask_positive, 'volume_trend_alignment'] = 1
    data.loc[mask_negative, 'volume_trend_alignment'] = 1
    data.loc[~mask_positive & ~mask_negative, 'volume_trend_alignment'] = -1
    
    # Volume spike adjustment
    data['volume_spike_adjustment'] = np.where(data['volume_spike_ratio'] > 1.2, 1.5, 
                                              np.where(data['volume_spike_ratio'] < 0.8, 0.7, 1.0))
    
    # Generate Combined Alpha Signal
    # Position momentum component
    position_momentum = data['position_change'] * (1 + 0.1 * data['position_change_consistency'])
    
    # Volume confirmation component
    volume_confirmation = data['volume_trend_alignment'] * data['volume_spike_adjustment']
    
    # Range momentum adjustment
    range_adjustment = 1 + data['range_change_pct'].fillna(0)
    
    # Final combined signal
    alpha_signal = position_momentum * volume_confirmation * range_adjustment
    
    # Clean up and return
    result = alpha_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
