import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price features
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Estimate session boundaries (simplified approach)
    # Morning session: first 2 hours, Afternoon session: last 2 hours
    # Using rolling windows to approximate session data
    data['high_morning'] = data['high'].rolling(window=2, min_periods=1).max()
    data['low_morning'] = data['low'].rolling(window=2, min_periods=1).min()
    data['close_morning'] = data['close'].rolling(window=2, min_periods=1).apply(lambda x: x[-1] if len(x) == 2 else x[0])
    
    data['high_afternoon'] = data['high'].rolling(window=2, min_periods=1).max()
    data['low_afternoon'] = data['low'].rolling(window=2, min_periods=1).min()
    
    # Estimate volume concentrations (simplified)
    data['volume_morning'] = data['volume'].rolling(window=2, min_periods=1).sum()
    data['volume_afternoon'] = data['volume'].rolling(window=2, min_periods=1).sum()
    
    # Midday price approximation
    data['midday_price'] = (data['open'] + data['close'].shift(1)) / 2
    
    # Morning Session Fracture Dynamics
    data['upper_fracture'] = data['high_morning'] - data['open']
    data['lower_fracture'] = data['open'] - data['low_morning']
    data['fracture_asymmetry'] = data['upper_fracture'] - data['lower_fracture']
    
    # Morning Volume Efficiency
    data['volume_concentration_morning'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[0]/x.sum() if len(x) == 2 and x.sum() > 0 else 0.5)
    data['fracture_momentum'] = data['fracture_asymmetry'] * data['volume_concentration_morning']
    
    # Afternoon Session Fracture Patterns
    data['upper_compression'] = data['high_afternoon'] - data['midday_price']
    data['lower_compression'] = data['midday_price'] - data['low_afternoon']
    data['compression_asymmetry'] = data['upper_compression'] - data['lower_compression']
    
    # Afternoon Volume Confirmation
    data['volume_concentration_afternoon'] = data['volume'].rolling(window=2, min_periods=1).apply(lambda x: x[-1]/x.sum() if len(x) == 2 and x.sum() > 0 else 0.5)
    data['compression_momentum'] = data['compression_asymmetry'] * data['volume_concentration_afternoon']
    
    # Session Transition Quality
    data['fracture_persistence'] = data['fracture_momentum'] * data['compression_momentum']
    data['transition_momentum'] = (data['close'] - data['midday_price']) * (data['midday_price'] - data['open'])
    
    data['session_consistency'] = np.sign(data['fracture_asymmetry']) * np.sign(data['compression_asymmetry'])
    data['volume_transition_efficiency'] = data['volume_concentration_morning'] * data['volume_concentration_afternoon']
    
    # Extreme Achievement Reversion
    data['volume_efficiency_highs'] = (data['amount'] / data['volume']) * (data['high'] - data['prev_high'])
    data['high_maintenance'] = data['high'] - data['open']
    
    data['volume_efficiency_lows'] = (data['amount'] / data['volume']) * (data['prev_low'] - data['low'])
    data['low_defense'] = data['open'] - data['low']
    
    data['high_reversion_efficiency'] = (data['high'] - data['close']) * data['volume_efficiency_highs']
    data['low_reversion_efficiency'] = (data['close'] - data['low']) * data['volume_efficiency_lows']
    data['extreme_asymmetry'] = data['high_reversion_efficiency'] - data['low_reversion_efficiency']
    
    data['large_trade_impact'] = (data['amount'] / data['volume']) * (data['high'] - data['low'])
    data['buy_sell_coordination'] = data['high_maintenance'] * data['low_defense']
    data['amount_concentration'] = data['large_trade_impact'] * data['buy_sell_coordination']
    
    # Fracture Divergence Detection
    data['morning_fracture_divergence'] = data['fracture_momentum'] / data['volume_concentration_morning'].replace(0, 1e-6)
    data['afternoon_compression_divergence'] = data['compression_momentum'] / data['volume_concentration_afternoon'].replace(0, 1e-6)
    data['cross_session_divergence'] = data['morning_fracture_divergence'] * data['afternoon_compression_divergence']
    
    data['high_low_divergence'] = data['high_reversion_efficiency'] - data['low_reversion_efficiency']
    data['volume_extreme_alignment'] = data['volume_efficiency_highs'] * data['volume_efficiency_lows']
    data['reversion_momentum_divergence'] = data['high_low_divergence'] * data['volume_extreme_alignment']
    
    data['boundary_extreme_alignment'] = data['cross_session_divergence'] * data['reversion_momentum_divergence']
    data['volume_confirmation'] = data['volume_concentration_morning'] * data['volume_concentration_afternoon'] * data['amount_concentration']
    data['fracture_consistency'] = data['session_consistency'] * data['boundary_extreme_alignment']
    
    # Momentum Integration
    data['morning_momentum'] = (data['close_morning'] - data['open']) * (data['high_morning'] - data['low_morning']) / data['high'].sub(data['low']).replace(0, 1e-6)
    data['afternoon_momentum'] = (data['close'] - data['midday_price']) * (data['high_afternoon'] - data['low_afternoon']) / data['high'].sub(data['low']).replace(0, 1e-6)
    
    data['session_momentum_divergence'] = data['morning_momentum'] * data['afternoon_momentum'] * data['cross_session_divergence']
    data['extreme_momentum_integration'] = data['transition_momentum'] * data['reversion_momentum_divergence']
    data['fracture_momentum_enhancement'] = data['session_momentum_divergence'] * data['extreme_momentum_integration']
    
    data['volume_efficiency_momentum'] = data['fracture_momentum_enhancement'] * data['volume_confirmation']
    data['amount_driven_momentum'] = data['volume_efficiency_momentum'] * data['amount_concentration']
    data['cross_session_momentum_validation'] = data['amount_driven_momentum'] * data['fracture_consistency']
    
    # Final Alpha Construction
    data['boundary_fracture_component'] = data['cross_session_divergence'] * (data['fracture_persistence'] + data['transition_momentum'] + data['session_consistency'] + data['volume_transition_efficiency'])
    data['extreme_reversion_component'] = data['reversion_momentum_divergence'] * (data['extreme_asymmetry'] + data['large_trade_impact'] + data['buy_sell_coordination'] + data['amount_concentration'])
    data['momentum_integration_component'] = data['cross_session_momentum_validation']
    
    data['fracture_reversion_alignment'] = data['boundary_fracture_component'] * data['extreme_reversion_component']
    data['volume_momentum_confirmation'] = data['fracture_reversion_alignment'] * data['momentum_integration_component']
    data['cross_session_validation'] = data['volume_momentum_confirmation'] * data['fracture_consistency']
    
    # Final Alpha Factor
    alpha = data['cross_session_validation'] * data['amount_driven_momentum']
    
    return alpha
