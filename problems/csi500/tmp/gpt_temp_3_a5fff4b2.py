import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['daily_range'] = data['high'] - data['low']
    data['range_efficiency'] = np.where(data['daily_range'] > 0, 
                                      (data['close'] - data['open']) / data['daily_range'], 0)
    
    # Intraday Divergence
    data['directional_div'] = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume'].shift(1))
    data['magnitude_div'] = data['range_efficiency'] * (data['volume'] / data['volume'].shift(1) - 1)
    
    # Multi-timeframe Divergence
    data['short_term_div'] = (data['close'] / data['close'].shift(5) - 1) * (data['volume'] / data['volume'].shift(5) - 1)
    data['medium_term_div'] = (data['close'] / data['close'].shift(20) - 1) * (data['volume'] / data['volume'].shift(20) - 1)
    
    # Range-Volume Divergence
    data['volume_efficiency'] = np.where(data['daily_range'] > 0, 
                                       data['volume'] / data['daily_range'], 0)
    
    # Divergence Persistence
    # Directional Persistence over 3 days
    data['directional_persistence'] = data['directional_div'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    # Multi-timeframe Alignment
    data['multi_timeframe_align'] = np.sign(data['short_term_div']) * np.sign(data['medium_term_div'])
    
    # Core Divergence - combine intraday and multi-timeframe signals
    data['core_divergence'] = (
        0.4 * data['directional_div'].fillna(0) +
        0.3 * data['magnitude_div'].fillna(0) +
        0.3 * (data['short_term_div'].fillna(0) + data['medium_term_div'].fillna(0)) / 2
    )
    
    # Enhanced Factor - Core Divergence × Persistence signals
    data['factor'] = (
        data['core_divergence'] * 
        (1 + 0.2 * data['directional_persistence']) * 
        (1 + 0.1 * data['multi_timeframe_align'])
    )
    
    # Return the factor series
    return data['factor']
