import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous close
    data['prev_close'] = data['close'].shift(1)
    
    # Calculate first hour data (assuming first hour is first 25% of trading day)
    data['first_hour_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x.max() if len(x) == 2 else np.nan)
    data['first_hour_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x.min() if len(x) == 2 else np.nan)
    data['first_hour_amount'] = data['amount'].rolling(window=2, min_periods=1).apply(lambda x: x.sum() * 0.25 if len(x) == 2 else np.nan)
    data['first_hour_volume'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x.sum() * 0.25 if len(x) == 2 else np.nan)
    
    # Calculate last hour data (assuming last hour is last 25% of trading day)
    data['last_hour_high'] = data['high'].rolling(window=2, min_periods=1).apply(lambda x: x.max() if len(x) == 2 else np.nan)
    data['last_hour_low'] = data['low'].rolling(window=2, min_periods=1).apply(lambda x: x.min() if len(x) == 2 else np.nan)
    data['last_hour_amount'] = data['amount'].rolling(window=2, min_periods=1).apply(lambda x: x.sum() * 0.25 if len(x) == 2 else np.nan)
    data['last_hour_volume'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x.sum() * 0.25 if len(x) == 2 else np.nan)
    
    # Opening Fractal Dynamics
    data['gap_fractal_momentum'] = ((data['open'] - data['prev_close']) * 
                                  (data['first_hour_amount'] / data['first_hour_volume']))
    
    data['opening_range_elasticity'] = (((data['high'] - data['open']) - (data['open'] - data['low'])) * 
                                      (data['amount'] / data['volume']))
    
    data['gap_position_efficiency'] = (((data['open'] - data['low']) / (data['high'] - data['low'])) * 
                                     (data['first_hour_amount'] / data['first_hour_volume']))
    
    # Intraday Flow Distribution Patterns
    data['morning_amount'] = data['first_hour_amount']
    data['afternoon_amount'] = data['amount'] - data['first_hour_amount']
    data['flow_skewness'] = (data['morning_amount'] - data['afternoon_amount']) / data['amount']
    
    data['volatility_expansion'] = (data['high'] - data['low']) / (data['open'] - data['prev_close']).abs()
    
    data['flow_persistence'] = data['amount'] / data['amount'].rolling(window=3, min_periods=1).mean()
    
    # Fractal Divergence Detection
    data['upper_fractal_flow'] = ((data['high'] - np.maximum(data['open'], data['close'])) * 
                                (data['amount'] / data['volume']))
    
    data['lower_fractal_flow'] = ((np.minimum(data['open'], data['close']) - data['low']) * 
                                (data['amount'] / data['volume']))
    
    data['net_fractal_divergence'] = data['upper_fractal_flow'] - data['lower_fractal_flow']
    
    # Session Transition Analysis
    data['morning_session_intensity'] = ((data['first_hour_high'] - data['first_hour_low']) * 
                                       (data['first_hour_amount'] / data['first_hour_volume']))
    
    data['afternoon_session_intensity'] = ((data['last_hour_high'] - data['last_hour_low']) * 
                                         (data['last_hour_amount'] / data['last_hour_volume']))
    
    data['session_flow_divergence'] = data['morning_session_intensity'] - data['afternoon_session_intensity']
    
    # Momentum Integration Framework
    data['opening_momentum'] = data['gap_fractal_momentum'] * data['opening_range_elasticity']
    
    data['flow_distribution_momentum'] = (data['flow_skewness'] * data['volatility_expansion'] * 
                                        data['flow_persistence'])
    
    data['fractal_divergence_momentum'] = data['net_fractal_divergence'] * data['session_flow_divergence']
    
    data['price_flow_efficiency'] = (data['close'] - data['open']).abs() / data['amount']
    
    # Final Alpha Integration
    data['core_momentum'] = data['opening_momentum'] * data['flow_distribution_momentum']
    data['divergence_signal'] = data['fractal_divergence_momentum'] * data['price_flow_efficiency']
    
    # Final factor calculation
    factor = data['core_momentum'] * data['divergence_signal'] * data['gap_position_efficiency']
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
