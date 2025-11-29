import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range using previous day's close
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Calculate Range Efficiency Ratio
    data['range_efficiency_ratio'] = (data['high'] - data['low']) / data['true_range']
    data['range_efficiency_ratio'] = data['range_efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate Range Efficiency Momentum
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['range_efficiency_momentum'] = (data['mid_price'] - data['prev_close']) * data['range_efficiency_ratio']
    
    # Calculate Volume Momentum
    data['volume_3d_median'] = data['volume'].rolling(window=3, min_periods=1).median()
    data['volume_momentum'] = data['volume'] / data['volume_3d_median']
    data['volume_momentum'] = data['volume_momentum'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate Volume-Range Divergence using 5-day rolling percentiles
    data['range_5d_pct'] = data['range_efficiency_ratio'].rolling(window=5, min_periods=1).apply(
        lambda x: (x.iloc[-1] - np.percentile(x, 25)) / (np.percentile(x, 75) - np.percentile(x, 25)) if len(x) >= 3 else np.nan
    )
    data['volume_5d_pct'] = data['volume_momentum'].rolling(window=5, min_periods=1).apply(
        lambda x: (x.iloc[-1] - np.percentile(x, 25)) / (np.percentile(x, 75) - np.percentile(x, 25)) if len(x) >= 3 else np.nan
    )
    data['volume_range_divergence'] = data['range_5d_pct'] - data['volume_5d_pct']
    
    # Calculate Volume Adjustment Factor scaled by 10-day rolling volatility
    data['range_volatility_10d'] = data['range_efficiency_ratio'].rolling(window=10, min_periods=3).std()
    data['volume_adjustment'] = np.tanh(data['volume_range_divergence'] / (data['range_volatility_10d'] + 1e-8))
    
    # Calculate volatility persistence ratio (5-day vs 20-day range std)
    data['range_std_5d'] = data['range_efficiency_ratio'].rolling(window=5, min_periods=3).std()
    data['range_std_20d'] = data['range_efficiency_ratio'].rolling(window=20, min_periods=10).std()
    data['volatility_persistence'] = data['range_std_5d'] / (data['range_std_20d'] + 1e-8)
    
    # Generate Composite Alpha Factor with conditional logic
    data['composite_factor'] = data['range_efficiency_momentum'] * (1 + data['volume_adjustment'])
    
    # Apply conditional logic based on divergence type
    bullish_condition = (data['volume_range_divergence'] > 0) & (data['range_efficiency_momentum'] > 0)
    bearish_condition = (data['volume_range_divergence'] < 0) & (data['range_efficiency_momentum'] < 0)
    
    data['final_factor'] = data['composite_factor']
    data.loc[bullish_condition, 'final_factor'] = data['composite_factor'] * 1.2
    data.loc[bearish_condition, 'final_factor'] = data['composite_factor'] * 0.8
    
    # Scale by volatility persistence ratio
    data['alpha_factor'] = data['final_factor'] * data['volatility_persistence']
    
    # Return the alpha factor series
    return data['alpha_factor']
