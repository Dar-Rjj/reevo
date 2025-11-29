import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday range position
    data['range_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    # Handle division by zero
    data['range_position'] = data['range_position'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate midpoint momentum
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['midpoint_momentum'] = data['midpoint'] - data['midpoint'].shift(1)
    
    # Calculate volume change
    data['volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Volume-weighted intraday signal
    data['volume_weighted_signal'] = data['range_position'] * data['volume']
    
    # Initialize factor column
    data['factor'] = 0.0
    
    # Calculate momentum-divergence signals
    for i in range(1, len(data)):
        if pd.notna(data.loc[data.index[i], 'midpoint_momentum']) and pd.notna(data.loc[data.index[i], 'volume_change']):
            midpoint_mom = data.loc[data.index[i], 'midpoint_momentum']
            vol_change = data.loc[data.index[i], 'volume_change']
            
            # Momentum-divergence classification
            if midpoint_mom > 0 and vol_change > 0:
                # Strong Bullish: Positive momentum with volume confirmation
                current_signal = 1.0
            elif midpoint_mom > 0 and vol_change <= 0:
                # Weak Bullish: Positive momentum without volume confirmation
                current_signal = 0.5
            elif midpoint_mom <= 0 and vol_change > 0:
                # Strong Bearish: Negative momentum with volume divergence
                current_signal = -1.0
            else:
                # Weak Bearish: Negative momentum without volume divergence
                current_signal = -0.5
            
            # Incorporate persistence from previous day
            if pd.notna(data.loc[data.index[i-1], 'factor']):
                persistence_weight = 0.3  # Weight for historical persistence
                data.loc[data.index[i], 'factor'] = (1 - persistence_weight) * current_signal + persistence_weight * data.loc[data.index[i-1], 'factor']
            else:
                data.loc[data.index[i], 'factor'] = current_signal
    
    # Fill initial NaN values with 0
    data['factor'] = data['factor'].fillna(0)
    
    return data['factor']
