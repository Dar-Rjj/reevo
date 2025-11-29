import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap Momentum Analysis
    # Calculate Overnight Gap Percentage
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Compute 5-day Return from Close
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    
    # Calculate Gap-Momentum Divergence
    data['gap_momentum_divergence'] = data['overnight_gap'] - data['return_5d']
    
    # Volatility Normalization
    # Calculate Daily Range (High - Low)
    data['daily_range'] = data['high'] - data['low']
    
    # Compute 5-day Average Range
    data['avg_range_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    
    # Divide Gap-Momentum Divergence by Average Range
    data['vol_adj_divergence'] = data['gap_momentum_divergence'] / data['avg_range_5d']
    
    # Volume Confirmation
    # Compute Volume Acceleration Factor
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_acceleration'] = data['volume_ma_5'] / data['volume_ma_10'] - 1
    
    # Assess Volume Concentration Timing
    data['volume_rank_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Composite Signal Generation
    # Combine Volatility-Adjusted Divergence and Volume Factors
    data['volume_confirmation'] = data['volume_acceleration'] * data['volume_rank_5d']
    
    # Apply Directional Multipliers
    data['directional_multiplier'] = np.where(
        data['vol_adj_divergence'] > 0, 
        np.where(data['volume_confirmation'] > 0, 1.5, 1.0),
        np.where(data['volume_confirmation'] < 0, -1.5, -1.0)
    )
    
    # Final composite factor
    data['factor'] = data['vol_adj_divergence'] * data['directional_multiplier'] * data['volume_confirmation']
    
    # Return the factor series
    return data['factor']
