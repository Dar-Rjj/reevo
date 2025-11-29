import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate variables
    data['prev_close'] = data['close'].shift(1)
    data['range'] = data['high'] - data['low']
    data['open_to_high'] = abs(data['high'] - data['open'])
    data['open_to_low'] = abs(data['low'] - data['open'])
    data['open_to_close'] = abs(data['close'] - data['open'])
    data['path_length'] = data['open_to_high'] + data['open_to_low'] + data['open_to_close']
    
    # Intraday Price Path Fractality
    data['fractal_dimension'] = np.where(data['range'] > 0, 
                                        np.log(data['path_length']) / np.log(data['range']), 
                                        1.0)
    
    # Volume-Weighted Acceleration Divergence
    data['vw_price'] = data['close'] * data['volume']
    data['vw_return'] = data['vw_price'].pct_change()
    data['vw_acceleration'] = data['vw_return'] - data['vw_return'].shift(1)
    
    data['price_return'] = data['close'].pct_change()
    data['price_acceleration'] = data['price_return'] - data['price_return'].shift(1)
    data['acceleration_divergence'] = data['vw_acceleration'] - data['price_acceleration']
    
    # Opening Gap Persistence
    data['gap'] = data['open'] - data['prev_close']
    data['gap_fill_ratio'] = np.where(
        data['gap'] > 0,
        (data['high'] - data['open']) / (data['high'] - data['prev_close']),
        (data['open'] - data['low']) / (data['prev_close'] - data['low'])
    )
    data['gap_fill_ratio'] = data['gap_fill_ratio'].fillna(0)
    
    # Gap persistence (rolling correlation between gap and intraday movement)
    data['intraday_move'] = data['close'] - data['open']
    gap_persistence = []
    for i in range(len(data)):
        if i < 10:
            gap_persistence.append(0)
            continue
        window_data = data.iloc[i-10:i]
        if len(window_data) >= 5:
            corr = window_data['gap'].corr(window_data['intraday_move'])
            gap_persistence.append(corr if not np.isnan(corr) else 0)
        else:
            gap_persistence.append(0)
    data['gap_persistence'] = gap_persistence
    
    # Amount-Volume Efficiency Momentum
    data['efficiency'] = data['amount'] / data['volume']
    data['efficiency_change'] = data['efficiency'].pct_change()
    data['price_momentum'] = data['close'].pct_change(3)
    data['efficiency_price_divergence'] = data['efficiency_change'] - data['price_momentum']
    
    # Intraday Range Expansion
    data['range_change'] = data['range'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    
    # Range-volume correlation (rolling)
    range_volume_corr = []
    for i in range(len(data)):
        if i < 15:
            range_volume_corr.append(0)
            continue
        window_data = data.iloc[i-15:i]
        if len(window_data) >= 8:
            corr = window_data['range_change'].corr(window_data['volume_change'])
            range_volume_corr.append(corr if not np.isnan(corr) else 0)
        else:
            range_volume_corr.append(0)
    data['range_volume_correlation'] = range_volume_corr
    
    # Combine factors with appropriate weights
    # Standardize each component
    components = [
        -data['fractal_dimension'],  # Lower fractality suggests more predictable patterns
        data['acceleration_divergence'],
        -data['gap_fill_ratio'],  # Less gap filling suggests stronger momentum
        data['gap_persistence'],
        data['efficiency_price_divergence'],
        data['range_volume_correlation']  # Positive correlation suggests genuine range expansion
    ]
    
    # Z-score normalization for each component
    normalized_components = []
    for component in components:
        mean_val = component.rolling(window=20, min_periods=10).mean()
        std_val = component.rolling(window=20, min_periods=10).std()
        normalized = (component - mean_val) / std_val
        normalized_components.append(normalized.fillna(0))
    
    # Equal weighted combination
    factor = sum(normalized_components) / len(normalized_components)
    
    return factor
