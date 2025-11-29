import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Breakout Signal
    # Calculate Current Volatility
    data['intraday_range'] = data['high'] - data['low']
    data['volatility_5d_avg'] = data['intraday_range'].rolling(window=5, min_periods=3).mean()
    data['volatility_ratio'] = data['intraday_range'] / data['volatility_5d_avg']
    
    # Breakout Detection
    data['volatility_3d_max'] = data['volatility_ratio'].rolling(window=3, min_periods=2).max()
    volatility_percentile = data['volatility_ratio'].rolling(window=10, min_periods=5).apply(
        lambda x: np.percentile(x.dropna(), 95) if len(x.dropna()) > 0 else np.nan, raw=False
    )
    data['volatility_breakout'] = (data['volatility_ratio'] > volatility_percentile).astype(float)
    
    # Momentum Efficiency Component
    # Calculate Intraday Momentum
    data['price_momentum'] = data['close'] - data['open']
    data['momentum_efficiency'] = data['price_momentum'] / data['intraday_range']
    data['momentum_efficiency'] = data['momentum_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Breakout Momentum Strength
    data['momentum_3d_avg'] = data['momentum_efficiency'].rolling(window=3, min_periods=2).mean()
    momentum_percentile = data['momentum_efficiency'].rolling(window=10, min_periods=5).apply(
        lambda x: np.percentile(x.dropna(), 90) if len(x.dropna()) > 0 else np.nan, raw=False
    )
    data['momentum_strength'] = (data['momentum_efficiency'] > momentum_percentile).astype(float)
    
    # Volume Confirmation
    # Volume Surge Detection
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / data['volume_5d_avg']
    data['volume_surge'] = (data['volume_ratio'] > 1.5).astype(float)
    
    # Amount Efficiency Component
    data['amount_per_volume'] = data['amount'] / data['volume']
    data['amount_per_volume'] = data['amount_per_volume'].replace([np.inf, -np.inf], np.nan)
    data['amount_efficiency_5d_avg'] = data['amount_per_volume'].rolling(window=5, min_periods=3).mean()
    data['amount_efficiency_ratio'] = data['amount_per_volume'] / data['amount_efficiency_5d_avg']
    
    # Combine Signals
    # Multiply Volatility Breakout Strength by Momentum Efficiency
    data['core_signal'] = data['volatility_breakout'] * data['momentum_efficiency']
    
    # Adjust by Volume Surge Magnitude and Amount Efficiency Ratio
    data['volume_adjustment'] = data['volume_surge'] * data['volume_ratio']
    data['amount_adjustment'] = data['amount_efficiency_ratio']
    
    # Final factor calculation
    data['factor'] = data['core_signal'] * data['volume_adjustment'] * data['amount_adjustment']
    
    return data['factor']
