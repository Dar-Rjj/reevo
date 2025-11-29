import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate intraday price extremes
    data['high_open_diff'] = data['high'] - data['open']
    data['low_open_diff'] = data['low'] - data['open']
    data['abs_high_open'] = np.abs(data['high_open_diff'])
    data['abs_low_open'] = np.abs(data['low_open_diff'])
    data['max_intraday_dev'] = data[['abs_high_open', 'abs_low_open']].max(axis=1)
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = np.abs(data['high'] - data['prev_close'])
    data['tr3'] = np.abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate 10-day average True Range
    data['avg_true_range'] = data['true_range'].rolling(window=10, min_periods=1).mean()
    
    # Calculate volatility-adjusted intraday magnitude
    data['vol_adj_intraday'] = data['max_intraday_dev'] / data['avg_true_range']
    
    # Calculate volume momentum components
    data['volume_5d_slope'] = data['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    data['volume_20d_slope'] = data['volume'].rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
    )
    
    # Calculate volume acceleration factor
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_accel'] = np.log(data['volume'] / data['prev_volume'])
    data['volume_accel'] = data['volume_accel'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate volatility slope components
    data['vol_adj_intraday_5d_slope'] = data['vol_adj_intraday'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
    )
    data['vol_adj_intraday_20d_slope'] = data['vol_adj_intraday'].rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
    )
    
    # Calculate volatility-volume divergence ratios
    data['divergence_5d'] = data['vol_adj_intraday_5d_slope'] / data['volume_5d_slope']
    data['divergence_20d'] = data['vol_adj_intraday_20d_slope'] / data['volume_20d_slope']
    
    # Calculate final factor
    data['factor'] = (data['divergence_5d'] + data['divergence_20d']) * data['volume_accel'] * data['vol_adj_intraday']
    
    # Clean up intermediate columns
    result = data['factor'].copy()
    
    return result
