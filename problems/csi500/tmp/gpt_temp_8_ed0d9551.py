import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Volatility Breakout Detection
    # Calculate True Range
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    data['TR'] = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # 10-day rolling average of TR
    data['TR_10d_avg'] = data['TR'].rolling(window=10, min_periods=1).mean()
    
    # Breakout ratio
    data['breakout_ratio'] = data['TR'] / data['TR_10d_avg']
    
    # Momentum Acceleration Analysis
    # Intraday Momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / data['TR'].replace(0, np.nan)
    
    # Momentum slope (current momentum - momentum 2 days ago) / 2
    data['momentum_slope'] = (data['intraday_momentum'] - data['intraday_momentum'].shift(2)) / 2
    
    # Acceleration Divergence Detection
    # Primary divergence: momentum slope vs intraday momentum sign mismatch
    data['primary_divergence'] = np.where(
        data['momentum_slope'] * data['intraday_momentum'] < 0, 
        abs(data['momentum_slope'] - data['intraday_momentum']), 
        0
    )
    
    # Secondary divergence: 3-day vs 10-day momentum direction mismatch
    momentum_3d = data['intraday_momentum'].rolling(window=3, min_periods=1).mean()
    momentum_10d = data['intraday_momentum'].rolling(window=10, min_periods=1).mean()
    data['secondary_divergence'] = np.where(
        momentum_3d * momentum_10d < 0, 
        abs(momentum_3d - momentum_10d), 
        0
    )
    
    # Combined divergence multiplier
    data['divergence_multiplier'] = 1 + data['primary_divergence'] + data['secondary_divergence']
    
    # Volume Acceleration Confirmation
    # Volume Ratio: current volume / 5-day average volume
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Volume Acceleration: daily volume change / 5-day volume mean
    data['volume_acceleration'] = data['volume'].diff() / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Signal Construction
    # Breakout Strength: breakout ratio × direction (Close vs (High+Low)/2)
    hl_midpoint = (data['high'] + data['low']) / 2
    data['breakout_direction'] = np.where(data['close'] > hl_midpoint, 1, -1)
    data['breakout_strength'] = data['breakout_ratio'] * data['breakout_direction']
    
    # Volume Multiplier
    data['volume_multiplier'] = data['volume_ratio'] * (1 + abs(data['volume_acceleration']))
    
    # Final Factor Generation
    # Base signal = breakout strength × momentum × divergence multiplier
    base_signal = data['breakout_strength'] * data['intraday_momentum'] * data['divergence_multiplier']
    
    # Apply volume confirmation
    final_factor = base_signal * data['volume_multiplier']
    
    # Sign adjustment based on momentum direction and extreme compression cases
    # Extreme compression: when TR is very low relative to its history
    tr_percentile = data['TR'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Adjust sign for momentum direction and compression
    momentum_direction = np.sign(data['intraday_momentum'])
    final_factor = final_factor * momentum_direction
    
    # For extreme compression cases, amplify the signal
    compression_multiplier = np.where(abs(tr_percentile) > 1, 1.5, 1.0)
    final_factor = final_factor * compression_multiplier
    
    return final_factor
