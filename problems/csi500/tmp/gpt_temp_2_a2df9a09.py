import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily price range
    data['daily_range'] = data['high'] - data['low']
    
    # Compute 5-day rolling average of daily price range
    data['avg_range_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    
    # Derive compression ratio: current range / 5-day average range
    data['compression_ratio'] = data['daily_range'] / data['avg_range_5d']
    
    # Flag compression when ratio < 0.7 (significant range narrowing)
    data['compression_flag'] = (data['compression_ratio'] < 0.7).astype(int)
    
    # Calculate daily turnover: Volume * Close
    data['daily_turnover'] = data['volume'] * data['close']
    
    # Compute 10-day rolling median of daily turnover
    data['median_turnover_10d'] = data['daily_turnover'].rolling(window=10, min_periods=5).median()
    
    # Calculate turnover deviation: (current turnover - 10-day median) / 10-day median
    data['turnover_deviation'] = (data['daily_turnover'] - data['median_turnover_10d']) / data['median_turnover_10d']
    
    # Identify breakout when deviation > 0.3 (substantial turnover increase)
    data['breakout_flag'] = (data['turnover_deviation'] > 0.3).astype(int)
    
    # Detect compression-before-breakout patterns
    # Look for compression in the previous 3 days before breakout
    data['compression_before_breakout'] = 0
    for i in range(3, len(data)):
        if data['breakout_flag'].iloc[i] == 1:
            # Check if there was compression in any of the previous 3 days
            if any(data['compression_flag'].iloc[i-3:i] == 1):
                data['compression_before_breakout'].iloc[i] = 1
    
    # Calculate divergence strength: compression ratio * turnover deviation
    data['divergence_strength'] = data['compression_ratio'] * data['turnover_deviation']
    
    # Apply time-decay weighting to recent compression signals
    # Create decay weights for compression signals (more recent = higher weight)
    decay_weights = np.array([0.1, 0.3, 0.6])  # weights for t-2, t-1, t
    data['weighted_compression'] = 0.0
    
    for i in range(2, len(data)):
        compression_window = data['compression_flag'].iloc[i-2:i+1].values
        weighted_sum = np.sum(compression_window * decay_weights)
        data['weighted_compression'].iloc[i] = weighted_sum
    
    # Generate factor predicting breakout continuation vs reversal
    # Factor combines divergence strength with weighted compression signals
    data['compression_breakout_factor'] = (
        data['divergence_strength'] * 
        (1 + data['weighted_compression']) * 
        data['compression_before_breakout']
    )
    
    # Return the factor series
    return data['compression_breakout_factor']
