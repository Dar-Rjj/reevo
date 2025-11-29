import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Intraday Range and Momentum Metrics
    data['daily_range'] = data['high'] - data['low']
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['rolling_median_midpoint'] = data['midpoint'].rolling(window=5, min_periods=1).median()
    data['momentum_signal'] = data['midpoint'] - data['rolling_median_midpoint']
    
    # Assess Range Persistence Characteristics
    data['range_percentile_rank'] = data['daily_range'].rolling(window=5, min_periods=1).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Calculate consecutive days with expanding range
    data['range_expanding'] = data['daily_range'] > data['daily_range'].shift(1)
    streak = 0
    streak_values = []
    for i, expanding in enumerate(data['range_expanding']):
        if i == 0 or not expanding:
            streak = 0
        else:
            streak += 1
        streak_values.append(streak)
    data['range_streak'] = streak_values
    data['range_streak_decay'] = 0.9 ** data['range_streak']
    
    # Analyze Volume-Weighted Confirmation Patterns
    data['high_volume_weighted'] = data['high'] * data['volume']
    data['low_volume_weighted'] = data['low'] * data['volume']
    data['extreme_ratio'] = data['high_volume_weighted'] / data['low_volume_weighted']
    
    # Calculate Volume Deviation Signal
    data['rolling_median_volume'] = data['volume'].rolling(window=20, min_periods=1).median()
    data['volume_deviation_signal'] = data['volume'] / data['rolling_median_volume']
    
    # Assess Volume Trend
    data['volume_trend'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_trend'] = data['volume_trend'].fillna(0)
    
    # Synthesize Composite Factor
    data['divergence_component'] = data['momentum_signal'] / data['daily_range']
    data['factor'] = (data['divergence_component'] * 
                     data['range_percentile_rank'] * 
                     data['extreme_ratio'] * 
                     data['volume_deviation_signal'] * 
                     data['volume_trend'])
    
    # Replace infinite values and NaN with 0
    data['factor'] = data['factor'].replace([np.inf, -np.inf], 0).fillna(0)
    
    return data['factor']
