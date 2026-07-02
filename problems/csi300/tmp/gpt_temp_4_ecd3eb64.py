import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate Momentum Reversal Component
    momentum_reversal = df['close'] / df['close'].shift(5)
    
    # Normalize Momentum Reversal
    momentum_zscore = momentum_reversal.rolling(window=20, min_periods=1).apply(
        lambda x: zscore(x, ddof=1)[-1] if len(x) > 1 else 0
    )
    normalized_momentum = np.clip(momentum_zscore, -3, 3) / 3  # Scale to [-1, 1]
    
    # Calculate Volume Context Component
    rolling_volume_avg = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_deviation = df['volume'] / rolling_volume_avg
    
    # Normalize Volume Deviation
    normalized_volume = np.clip(np.abs(volume_deviation), 0, 2)  # Scale to [0, 2]
    
    # Signal Generation
    combined_signal = normalized_momentum * normalized_volume
    final_signal = combined_signal * df['open'] / df['volume']
    
    return final_signal
