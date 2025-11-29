import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Opening Gap Components
    data['prev_close'] = data['close'].shift(1)
    data['gap_magnitude'] = data['open'] - data['prev_close']
    data['abs_gap'] = np.abs(data['gap_magnitude'])
    data['gap_direction'] = np.sign(data['gap_magnitude'])
    
    # Calculate Morning Range and Efficiency
    data['morning_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x[0])
    data['morning_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x[0])
    data['morning_range'] = data['morning_high'] - data['morning_low']
    
    # Calculate Morning Turnover Density
    data['morning_amount'] = data['amount'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x[0])
    data['morning_volume'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0] if len(x) == 2 else x[0])
    data['morning_efficiency'] = data['morning_amount'] / (data['morning_volume'] * data['morning_range'].replace(0, np.nan))
    
    # Calculate Afternoon Range and Efficiency
    data['afternoon_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else x[0])
    data['afternoon_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else x[0])
    data['afternoon_range'] = data['afternoon_high'] - data['afternoon_low']
    
    # Calculate Afternoon Turnover Density
    data['afternoon_amount'] = data['amount'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else x[0])
    data['afternoon_volume'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[1] if len(x) == 2 else x[0])
    data['afternoon_efficiency'] = data['afternoon_amount'] / (data['afternoon_volume'] * data['afternoon_range'].replace(0, np.nan))
    
    # Compute Efficiency Divergence Factor
    data['efficiency_ratio'] = data['morning_efficiency'] / data['afternoon_efficiency'].replace(0, np.nan)
    data['efficiency_divergence'] = data['efficiency_ratio'] * data['gap_direction']
    data['scaled_divergence'] = data['efficiency_divergence'] * data['abs_gap']
    data['sqrt_divergence'] = np.sqrt(np.abs(data['scaled_divergence'])) * np.sign(data['scaled_divergence'])
    
    # Calculate Intraday Volatility Structure
    data['morning_volatility'] = (data['morning_high'] - data['morning_low']) / data['open'].replace(0, np.nan)
    data['afternoon_volatility'] = (data['afternoon_high'] - data['afternoon_low']) / data['close'].replace(0, np.nan)
    data['volatility_ratio'] = data['morning_volatility'] / data['afternoon_volatility'].replace(0, np.nan)
    
    # Combine with Efficiency Divergence
    data['vol_weighted_divergence'] = data['sqrt_divergence'] * data['volatility_ratio']
    
    # Apply Volume Persistence
    data['volume_trend'] = data['volume'].rolling(window=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and np.std(x) > 0 else 0)
    data['volume_persistence'] = np.abs(data['volume_trend'])
    
    # Final factor calculation
    factor = data['vol_weighted_divergence'] * data['volume_persistence']
    
    return factor
