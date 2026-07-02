import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Initialize output series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate daily returns
    daily_returns = data['close'].pct_change()
    
    # 1. Measure Price Momentum
    # 5-day rolling sum of returns
    momentum_5d = daily_returns.rolling(5).sum()
    # Calculate 2*STD of momentum using expanding window (historical only)
    momentum_std = momentum_5d.expanding().std() * 2
    
    # Identify momentum surge (current > 2*STD and minimum 3 consecutive days)
    momentum_surge = (momentum_5d > momentum_std).astype(int)
    # Check for at least 3 consecutive days
    for i in range(2, len(momentum_surge)):
        if momentum_surge.iloc[i] == 1 and momentum_surge.iloc[i-1] == 1 and momentum_surge.iloc[i-2] == 1:
            momentum_surge.iloc[i] = 1
        else:
            momentum_surge.iloc[i] = 0
    
    # 2. Volume Confirmation Signal
    # Calculate 15-day MA of volume
    volume_ma_15d = data['volume'].rolling(15).mean()
    # Volume ratio (current / MA)
    volume_ratio = data['volume'] / volume_ma_15d
    # Threshold crossing (ratio > 2.0 and first occurrence in 7 days)
    volume_signal = (volume_ratio > 2.0).astype(int)
    # Ensure first occurrence in 7 days
    for i in range(1, len(volume_signal)):
        if volume_signal.iloc[i] == 1 and volume_signal.iloc[max(0, i-7):i].sum() == 0:
            volume_signal.iloc[i] = 1
        else:
            volume_signal.iloc[i] = 0
    
    # 3. Momentum Direction
    # Calculate price range component
    price_range = data['high'] - data['low']
    # Upward momentum condition
    upward = (data['close'] > (data['open'] + 0.3 * price_range)).astype(int)
    # Downward momentum condition
    downward = (data['close'] < (data['open'] - 0.3 * price_range)).astype(int) * -1
    # Combined direction signal
    direction_signal = upward + downward
    
    # Combine all signals
    factor = momentum_surge * volume_signal * direction_signal
    
    return factor
