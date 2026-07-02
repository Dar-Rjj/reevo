import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Trend Acceleration Component
    # Second Derivative
    ema10 = df['close'].ewm(span=10, adjust=False).mean()
    ema20 = df['close'].ewm(span=20, adjust=False).mean()
    
    delta_ema10 = ema10.diff(1)
    delta_ema20 = ema20.diff(1)
    
    second_derivative = delta_ema10 - delta_ema20
    
    # Normalization
    rolling_std = df['close'].rolling(window=10).std()
    normalized_acceleration = second_derivative / (rolling_std + 1e-6)
    
    # Liquidity Weighting
    # Volume-Adjusted
    rolling_mean_volume = df['volume'].rolling(window=20).mean()
    normalized_volume = df['volume'] / (rolling_mean_volume + 1e-6)
    
    # Microstructure Signal
    price_range = df['close'] - df['open']
    microstructure_zscore = price_range.rolling(window=5).apply(
        lambda x: (x[-1] - x[:-1].mean()) / (x[:-1].std() + 1e-6)
    )
    
    # Combine components
    weighted_acceleration = normalized_acceleration * normalized_volume * microstructure_zscore
    
    # Decay Factor
    decayed_signal = weighted_acceleration.ewm(alpha=0.3, adjust=False).mean()
    
    return decayed_signal
