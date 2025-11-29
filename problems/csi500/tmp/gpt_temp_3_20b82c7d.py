import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining price efficiency momentum and gap persistence momentum
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price Efficiency Momentum
    # Daily Efficiency Ratio
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Multi-timeframe efficiency trends
    data['eff_short_term'] = data['daily_efficiency'].rolling(window=3, min_periods=2).mean()
    data['eff_medium_term'] = data['daily_efficiency'].rolling(window=10, min_periods=5).mean()
    
    # Efficiency Breakout Signal
    data['efficiency_divergence'] = data['eff_short_term'] - data['eff_medium_term']
    
    # Volume Confirmation
    data['volume_ratio'] = data['volume'].rolling(window=5, min_periods=3).mean() / \
                          data['volume'].rolling(window=10, min_periods=5).mean()
    
    # 2. Gap Persistence Momentum
    # Opening Gap Analysis
    data['prev_close'] = data['close'].shift(1)
    data['gap_size'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    
    # Gap Direction Classification
    data['gap_direction'] = np.where(data['gap_size'] > 0, 1, 
                                   np.where(data['gap_size'] < 0, -1, 0))
    
    # Intraday Gap Behavior - Gap Filling Detection
    data['gap_filled_up'] = np.where((data['gap_direction'] == 1) & 
                                   (data['low'] <= data['prev_close']), 1, 0)
    data['gap_filled_down'] = np.where((data['gap_direction'] == -1) & 
                                     (data['high'] >= data['prev_close']), 1, 0)
    
    # Persistence Score
    data['gap_persistence'] = np.where(data['gap_filled_up'] | data['gap_filled_down'], -1, 1)
    data['gap_persistence'] = data['gap_persistence'] * np.abs(data['gap_size'])
    
    # 3. Composite Alpha Signal
    # Combine Efficiency Breakout with Gap Persistence
    data['composite_signal'] = (data['efficiency_divergence'] * 0.6 + 
                               data['gap_persistence'] * 0.4)
    
    # Volume-weighted signal
    data['volume_weight'] = data['volume_ratio'].rolling(window=5, min_periods=3).mean()
    data['weighted_signal'] = data['composite_signal'] * data['volume_weight']
    
    # Cross-sectional ranking (z-score normalization within each day)
    def cross_sectional_rank(group):
        if len(group) > 1:
            return (group - group.mean()) / (group.std() + 1e-8)
        else:
            return group * 0  # Return zeros if only one stock
    
    # Apply cross-sectional normalization
    alpha_factor = data.groupby(data.index)['weighted_signal'].transform(cross_sectional_rank)
    
    return alpha_factor
