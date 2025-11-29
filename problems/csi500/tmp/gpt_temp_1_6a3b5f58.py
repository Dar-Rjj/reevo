import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous day volume for reference
    data['prev_volume'] = data['volume'].shift(1)
    
    # Morning session calculations (assuming first half of trading day)
    data['morning_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 1 else max(x[0], x[1] * 0.5))
    data['morning_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 1 else min(x[0], x[1] * 0.5))
    
    # Estimate morning volume (assuming 50% of daily volume in morning)
    data['morning_volume'] = data['volume'] * 0.5
    
    # Afternoon session calculations (assuming second half of trading day)
    data['afternoon_high'] = data['high']
    data['afternoon_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[-1])
    
    # Estimate afternoon volume (remaining 50% of daily volume)
    data['afternoon_volume'] = data['volume'] * 0.5
    
    # Morning Acceleration Component
    data['morning_hl_range'] = data['morning_high'] - data['morning_low']
    data['morning_up_momentum'] = (data['morning_high'] - data['open']) / data['open']
    data['morning_down_momentum'] = (data['open'] - data['morning_low']) / data['open']
    data['net_morning_accel'] = data['morning_up_momentum'] - data['morning_down_momentum']
    
    # Afternoon Acceleration Component
    data['afternoon_hl_range'] = data['afternoon_high'] - data['afternoon_low']
    data['afternoon_up_momentum'] = (data['close'] - data['afternoon_low']) / data['afternoon_low']
    data['afternoon_down_momentum'] = (data['afternoon_high'] - data['close']) / data['afternoon_high']
    data['net_afternoon_accel'] = data['afternoon_up_momentum'] - data['afternoon_down_momentum']
    
    # Volatility-Efficient Adjustment
    data['morning_range_eff'] = (data['close'] - data['open']) / data['morning_hl_range'].replace(0, np.nan)
    data['afternoon_range_eff'] = (data['close'] - data['afternoon_low']) / data['afternoon_hl_range'].replace(0, np.nan)
    
    # Handle division by zero cases
    data['morning_range_eff'] = data['morning_range_eff'].fillna(0)
    data['afternoon_range_eff'] = data['afternoon_range_eff'].fillna(0)
    
    data['volatility_efficient_accel'] = (data['net_morning_accel'] * data['morning_range_eff'] + 
                                         data['net_afternoon_accel'] * data['afternoon_range_eff'])
    
    # Volume Intensity Analysis
    data['morning_volume_intensity'] = data['morning_volume'] / (data['prev_volume'] + data['morning_volume']).replace(0, np.nan)
    data['afternoon_volume_divergence'] = (data['afternoon_volume'] / data['morning_volume'] - 1).replace([np.inf, -np.inf], 0)
    data['volume_momentum'] = data['afternoon_volume'] / data['prev_volume'].replace(0, np.nan)
    
    # Handle division by zero cases
    data['morning_volume_intensity'] = data['morning_volume_intensity'].fillna(0.5)
    data['volume_momentum'] = data['volume_momentum'].fillna(1)
    
    # Price-Volume Alignment
    data['morning_pv_alignment'] = np.sign(data['net_morning_accel']) * np.sign(data['morning_volume_intensity'])
    data['afternoon_pv_alignment'] = np.sign(data['net_afternoon_accel']) * np.sign(data['afternoon_volume_divergence'])
    data['volume_price_confirmation'] = data['morning_pv_alignment'] * data['afternoon_pv_alignment']
    
    # Composite Factor Construction
    data['core_accel_signal'] = data['volatility_efficient_accel'] * data['volume_price_confirmation']
    data['volume_momentum_enhancement'] = data['core_accel_signal'] * data['volume_momentum']
    
    # Intraday Persistence Filter
    data['intraday_persistence'] = np.sign(data['net_morning_accel'] * data['net_afternoon_accel'])
    data['final_factor'] = data['volume_momentum_enhancement'] * data['intraday_persistence']
    
    # Return the factor series
    return data['final_factor']
