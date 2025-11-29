import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Range Analysis
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_magnitude'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    
    # Overnight Gap Analysis
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_abs'] = np.abs(data['overnight_gap'])
    data['gap_direction'] = np.sign(data['overnight_gap'])
    
    # Volume-Integrated Momentum
    data['range_to_gap_ratio'] = data['daily_range'] / (data['gap_abs'] + 1e-8)
    data['volume_rank'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Momentum scoring based on volume and gap characteristics
    data['volume_gap_score'] = np.where(
        (data['volume_rank'] > 0.7) & (data['gap_abs'] < 0.01),
        data['gap_direction'],  # High volume, small gaps → continuation
        np.where(
            (data['volume_rank'] < 0.3) & (data['gap_abs'] > 0.02),
            -data['gap_direction'],  # Low volume, large gaps → reversal
            0  # Neutral cases
        )
    )
    
    # Combine components into final factor
    data['momentum_decay_factor'] = (
        data['range_to_gap_ratio'] * data['volume_gap_score'] * 
        (1 - data['range_magnitude'])  # Decay component based on range magnitude
    )
    
    # Apply rolling normalization
    factor = data['momentum_decay_factor'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    )
    
    return factor
