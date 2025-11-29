import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling metrics for complexity analysis
    data['close_diff_1'] = data['close'].diff()
    data['close_diff_5'] = data['close'].diff(5)
    data['close_diff_10'] = data['close'].diff(10)
    
    # Multi-scale volatility ratio (5-day)
    data['vol_ratio_5'] = data['close_diff_1'].abs().rolling(window=5).sum() / data['close_diff_5'].abs().replace(0, np.nan)
    
    # Multi-scale volatility ratio (10-day)
    data['vol_ratio_10'] = data['close_diff_1'].abs().rolling(window=10).sum() / data['close_diff_10'].abs().replace(0, np.nan)
    
    # Complexity divergence
    data['complexity_div'] = data['vol_ratio_10'] - data['vol_ratio_5']
    
    # Price Range Utilization
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_range_efficiency'] = (data['close'] - data['open']).abs() / data['intraday_range'].replace(0, np.nan)
    
    # Overnight gap persistence
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)).abs()
    data['prev_day_range'] = (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    data['gap_persistence'] = data['overnight_gap'] / data['prev_day_range']
    
    # Calculate daily factor values
    for date in data.index:
        current_data = data.loc[:date].copy()
        
        if len(current_data) < 10:  # Need sufficient history
            factor.loc[date] = np.nan
            continue
        
        # Get current day data
        current_day = current_data.loc[date]
        
        # Complexity factors (using rolling windows)
        complexity_5 = current_data['vol_ratio_5'].iloc[-1] if not pd.isna(current_data['vol_ratio_5'].iloc[-1]) else 0
        complexity_div = current_data['complexity_div'].iloc[-1] if not pd.isna(current_data['complexity_div'].iloc[-1]) else 0
        
        # Price range factors
        range_efficiency = current_day['intraday_range_efficiency'] if not pd.isna(current_day['intraday_range_efficiency']) else 0
        gap_persistence = current_day['gap_persistence'] if not pd.isna(current_day['gap_persistence']) else 0
        
        # Volume factors (simplified - using total volume as proxy)
        volume_total = current_day['volume'] if not pd.isna(current_day['volume']) else 0
        
        # Flow efficiency composite
        flow_efficiency = range_efficiency * volume_total if not pd.isna(range_efficiency) and not pd.isna(volume_total) else 0
        
        # Combine factors (equal weighting for demonstration)
        factor_value = (
            complexity_5 + 
            complexity_div + 
            range_efficiency + 
            gap_persistence + 
            flow_efficiency
        )
        
        factor.loc[date] = factor_value
    
    # Handle any remaining NaN values
    factor = factor.fillna(0)
    
    return factor
