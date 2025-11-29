import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Midday Price
    data['midday'] = (data['high'] + data['low']) / 2
    
    # Calculate Price Acceleration
    denominator = np.abs(data['midday'] - data['open'])
    # Avoid division by zero
    denominator = denominator.replace(0, np.nan)
    data['acceleration'] = (data['close'] - data['midday']) / denominator
    
    # Calculate Price Range Ratio
    data['range_ratio'] = (data['high'] - data['low']) / data['open']
    
    # Calculate Trend Duration - count consecutive same-direction closes
    data['price_change'] = np.sign(data['close'] - data['close'].shift(1))
    data['trend_duration'] = 0
    
    # Calculate consecutive same direction periods
    for i in range(1, len(data)):
        if data['price_change'].iloc[i] == data['price_change'].iloc[i-1] and not pd.isna(data['price_change'].iloc[i]):
            data['trend_duration'].iloc[i] = data['trend_duration'].iloc[i-1] + 1
        else:
            data['trend_duration'].iloc[i] = 1
    
    # Calculate Volume Intensity
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_intensity'] = data['volume'] / data['volume_ma_10']
    
    # Calculate Volume-Price Alignment
    data['volume_price_alignment'] = np.sign(data['close'] - data['open']) * data['volume_intensity']
    
    # Combine components
    # Multiply Acceleration by Trend Duration
    data['momentum_component'] = data['acceleration'] * data['trend_duration']
    
    # Adjust by Volume-Price Alignment
    data['combined_factor'] = data['momentum_component'] * data['volume_price_alignment']
    
    # Apply damping for extreme values using tanh
    data['factor'] = np.tanh(data['combined_factor'] * 0.1)
    
    # Handle any remaining NaN values
    data['factor'] = data['factor'].fillna(0)
    
    return data['factor']
