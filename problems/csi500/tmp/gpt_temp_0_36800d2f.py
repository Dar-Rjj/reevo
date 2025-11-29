import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous day's close
    data['close_prev'] = data['close'].shift(1)
    
    # Gap & Volatility Analysis
    data['overnight_gap'] = (data['open'] - data['close_prev']) / data['close_prev']
    data['parkinson_vol'] = (data['high'] - data['low']) / data['close_prev']
    data['vol_regime'] = data['parkinson_vol'] / data['parkinson_vol'].rolling(window=20, min_periods=10).mean()
    
    # Price Momentum Integration
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['gap_momentum_alignment'] = np.sign(data['overnight_gap']) * np.sign(data['intraday_momentum'])
    
    # Volume Dynamics
    data['volume_prev'] = data['volume'].shift(1)
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'])
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    data['volume_confirmation'] = np.sign(data['overnight_gap']) * np.sign(data['volume'] - data['volume_prev'])
    
    # Core Signal
    data['core_signal'] = data['overnight_gap'] * data['vol_regime'] * data['intraday_momentum']
    
    # Volume Adjustment
    data['volume_adjustment'] = data['core_signal'] * data['volume_efficiency']
    
    # Regime Multipliers
    data['regime_multiplier'] = np.where(data['vol_regime'] > 1.0, 1.5, 0.7)
    
    # Final Factor
    data['gap_vol_momentum'] = data['core_signal'] * data['volume_adjustment'] * data['regime_multiplier']
    
    # Cross-sectional ranking (z-score normalization)
    factor = data.groupby(data.index)['gap_vol_momentum'].transform(lambda x: (x - x.mean()) / x.std())
    
    return factor
