import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Alpha Factor: Cross-Session Volatility Momentum Divergence
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Morning session calculations (assuming first half of trading day)
    data['morning_high'] = data['high'].rolling(window=2, min_periods=2).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    data['morning_low'] = data['low'].rolling(window=2, min_periods=2).apply(lambda x: x[0] if len(x) == 2 else np.nan)
    
    # Morning volatility (range relative to open)
    data['morning_range'] = (data['morning_high'] - data['morning_low']) / data['open']
    data['morning_volatility'] = data['morning_range'].rolling(window=5, min_periods=3).mean()
    
    # Afternoon session calculations (assuming second half of trading day)
    data['afternoon_high'] = data['high'].rolling(window=2, min_periods=2).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    data['afternoon_low'] = data['low'].rolling(window=2, min_periods=2).apply(lambda x: x[1] if len(x) == 2 else np.nan)
    
    # Afternoon volatility (range relative to session start - approximated by morning high/low midpoint)
    data['session_start'] = (data['morning_high'] + data['morning_low']) / 2
    data['afternoon_range'] = (data['afternoon_high'] - data['afternoon_low']) / data['session_start']
    data['afternoon_volatility'] = data['afternoon_range'].rolling(window=5, min_periods=3).mean()
    
    # Volatility regime comparison
    data['volatility_ratio'] = data['morning_volatility'] / (data['afternoon_volatility'] + 1e-8)
    data['volatility_diff'] = abs(data['morning_volatility'] - data['afternoon_volatility'])
    
    # Regime persistence
    data['high_vol_persistence'] = ((data['morning_volatility'] > data['morning_volatility'].rolling(window=10).mean()) & 
                                   (data['afternoon_volatility'] > data['afternoon_volatility'].rolling(window=10).mean())).astype(int)
    data['low_vol_persistence'] = ((data['morning_volatility'] < data['morning_volatility'].rolling(window=10).mean()) & 
                                  (data['afternoon_volatility'] < data['afternoon_volatility'].rolling(window=10).mean())).astype(int)
    
    # Momentum calculations
    data['morning_return'] = (data['session_start'] - data['open']) / data['open']
    data['afternoon_return'] = (data['close'] - data['session_start']) / data['session_start']
    
    # Session direction consistency
    data['direction_consistency'] = np.sign(data['morning_return']) * np.sign(data['afternoon_return'])
    
    # Volatility-weighted momentum
    data['morning_momentum_weighted'] = data['morning_return'] * data['volatility_ratio']
    data['afternoon_momentum_weighted'] = data['afternoon_return'] / (data['volatility_ratio'] + 1e-8)
    
    # Adjust for regime persistence
    regime_adjustment = np.where(data['high_vol_persistence'] == 1, 1.2, 
                                np.where(data['low_vol_persistence'] == 1, 0.8, 1.0))
    
    # Final divergence signal
    data['momentum_divergence'] = (data['morning_momentum_weighted'] - data['afternoon_momentum_weighted']) * regime_adjustment
    
    # Multi-period divergence (3-day momentum)
    data['divergence_3d'] = data['momentum_divergence'].rolling(window=3, min_periods=2).mean()
    
    # Final factor: Cross-session volatility momentum divergence
    factor = data['divergence_3d']
    
    return factor
