import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily price range
    data['daily_range'] = data['high'] - data['low']
    
    # Short-term Range Persistence (5-day autocorrelation)
    data['range_lag1'] = data['daily_range'].shift(1)
    data['range_lag2'] = data['daily_range'].shift(2)
    data['range_lag3'] = data['daily_range'].shift(3)
    data['range_lag4'] = data['daily_range'].shift(4)
    
    # Calculate rolling correlation between current range and lagged ranges
    range_corr_window = 20
    range_persistence = []
    for i in range(len(data)):
        if i < range_corr_window:
            range_persistence.append(np.nan)
        else:
            current_range = data['daily_range'].iloc[i-range_corr_window+1:i+1]
            lagged_range = data['range_lag1'].iloc[i-range_corr_window+1:i+1]
            corr = current_range.corr(lagged_range)
            range_persistence.append(corr if not np.isnan(corr) else 0)
    
    data['range_persistence'] = range_persistence
    
    # Long-term Range Breakout (compare to 60-day baseline)
    data['range_ma_60'] = data['daily_range'].rolling(window=60, min_periods=30).mean()
    data['range_breakout'] = data['daily_range'] / data['range_ma_60'] - 1
    
    # High-Low Momentum Spread
    data['high_momentum'] = data['high'] / data['high'].shift(1) - 1
    data['low_momentum'] = data['low'] / data['low'].shift(1) - 1
    data['momentum_spread'] = data['high_momentum'] - data['low_momentum']
    
    # Range-Normalized Divergence
    data['range_normalized_divergence'] = data['momentum_spread'] / (data['daily_range'] / data['close'])
    
    # Volume Trend Confirmation
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    
    # Volume-weighted moving average of volume ratio (5-day)
    data['volume_ratio_ma'] = data['volume_ratio'].rolling(window=5, min_periods=3).mean()
    
    # Combine range components
    data['range_divergence'] = data['range_persistence'] * data['range_breakout']
    
    # Final Factor Construction
    data['factor'] = (data['range_divergence'] * data['range_normalized_divergence'] * 
                     data['volume_ratio_ma'])
    
    # Clean up intermediate columns
    result = data['factor'].copy()
    
    return result
