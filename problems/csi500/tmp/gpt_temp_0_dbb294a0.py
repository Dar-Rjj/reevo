import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['range'] = data['high'] - data['low']
    data['prev_range'] = data['range'].shift(1)
    data['range_momentum'] = data['range'] - data['prev_range']
    
    # Opening gap analysis
    data['prev_close'] = data['close'].shift(1)
    data['gap_magnitude'] = data['open'] - data['prev_close']
    data['gap_direction'] = np.sign(data['gap_magnitude'])
    
    # Volume efficiency patterns
    data['price_change_per_volume'] = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    
    # Volume-range alignment quality
    data['volume_range_alignment'] = np.where(
        (data['range_momentum'] > 0) & (data['volume'] > data['volume'].shift(1)),
        1.0,
        np.where(
            (data['range_momentum'] < 0) & (data['volume'] < data['volume'].shift(1)),
            1.0,
            0.0
        )
    )
    
    # Amount flow analysis
    data['amount_flow_intensity'] = data['amount'] / (data['volume'] + 1e-8)
    data['price_impact_per_dollar'] = (data['close'] - data['open']) / (data['amount'] + 1e-8)
    
    # Flow-range timing patterns
    data['flow_range_timing'] = np.where(
        (data['range_momentum'] > 0) & (data['amount'] > data['amount'].shift(1)),
        1.0,
        np.where(
            (data['range_momentum'] < 0) & (data['amount'] < data['amount'].shift(1)),
            1.0,
            0.0
        )
    )
    
    # Gap efficiency integration
    data['gap_range_alignment'] = np.where(
        data['gap_direction'] * data['range_momentum'] > 0,
        1.0,
        -1.0
    )
    
    # Multi-day persistence assessment (using rolling windows)
    data['range_momentum_3d'] = data['range_momentum'].rolling(window=3, min_periods=1).mean()
    data['volume_alignment_3d'] = data['volume_range_alignment'].rolling(window=3, min_periods=1).mean()
    data['flow_timing_3d'] = data['flow_range_timing'].rolling(window=3, min_periods=1).mean()
    
    # Composite efficiency score components
    data['momentum_confirmation'] = (
        data['range_momentum'] * data['volume_range_alignment'] * 
        data['gap_range_alignment'] * data['flow_range_timing']
    )
    
    data['efficiency_strength'] = (
        data['price_change_per_volume'].abs() * 
        data['price_impact_per_dollar'].abs() *
        data['amount_flow_intensity']
    )
    
    # Final composite factor
    data['composite_efficiency'] = (
        data['momentum_confirmation'] * 
        data['efficiency_strength'] *
        data['range_momentum_3d'] *
        data['volume_alignment_3d'] *
        data['flow_timing_3d']
    )
    
    # Cross-sectional ranking (z-score normalization within each day)
    def cross_sectional_rank(group):
        if len(group) > 1:
            return (group - group.mean()) / (group.std() + 1e-8)
        else:
            return group * 0  # Return zeros if only one stock
    
    # Apply cross-sectional normalization
    factor = data.groupby(data.index)['composite_efficiency'].transform(cross_sectional_rank)
    
    return factor
