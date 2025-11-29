import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Remove first row with NaN values from shift
    data = data.dropna()
    
    # Opening Range Dynamics
    data['volatility_adjusted_opening_gap'] = (data['high'] - data['low']) / (abs(data['open'] - data['prev_close']).replace(0, np.nan))
    data['opening_position_strength'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['opening_wick_rejection'] = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Midday Range Patterns
    data['range_compression'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low']).replace(0, np.nan)
    data['volume_range_efficiency'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['prev_range_position'] = (data['prev_close'] - data['prev_low']) / (data['prev_high'] - data['prev_low']).replace(0, np.nan)
    data['range_position_momentum'] = data['opening_position_strength'] - data['prev_range_position']
    
    # Closing Range Quality
    data['range_expansion'] = (data['high'] - data['low']) - (data['prev_high'] - data['prev_low'])
    data['closing_position'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['final_hour_efficiency'] = data['closing_position'] * data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume-Range Coherence
    data['volume_range_alignment'] = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    data['amount_range_impact'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['volume_range_divergence'] = data['volume_range_alignment'] * data['range_expansion']
    
    # Range Movement Quality
    data['range_trend'] = (data['high'] - data['low']) / (data['prev_high'] - data['prev_low']).replace(0, np.nan)
    data['range_reversal'] = data['range_expansion'] * data['range_position_momentum']
    data['range_efficiency'] = data['volatility_adjusted_opening_gap'] * data['closing_position']
    
    # Cross-Sectional Range Signals
    data['range_performance'] = data['range_trend'] * data['closing_position']
    data['volume_range_leadership'] = data['range_expansion'] * data['volume_range_alignment']
    data['range_information_speed'] = data['volatility_adjusted_opening_gap'] * data['volume_range_alignment']
    
    # Combine all components into final factor
    factor_components = [
        'volatility_adjusted_opening_gap',
        'opening_position_strength', 
        'opening_wick_rejection',
        'range_compression',
        'volume_range_efficiency',
        'range_position_momentum',
        'range_expansion',
        'closing_position',
        'final_hour_efficiency',
        'volume_range_alignment',
        'amount_range_impact',
        'volume_range_divergence',
        'range_trend',
        'range_reversal',
        'range_efficiency',
        'range_performance',
        'volume_range_leadership',
        'range_information_speed'
    ]
    
    # Calculate z-scores for each component and average them
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for date in data.index:
        day_data = data.loc[date]
        valid_components = []
        
        for component in factor_components:
            if pd.notna(day_data[component]):
                valid_components.append(day_data[component])
        
        if valid_components:
            # Calculate cross-sectional z-score for the day
            component_mean = np.mean(valid_components)
            component_std = np.std(valid_components)
            if component_std > 0:
                factor_values[date] = (day_data[component] - component_mean) / component_std
            else:
                factor_values[date] = 0
        else:
            factor_values[date] = 0
    
    return factor_values
