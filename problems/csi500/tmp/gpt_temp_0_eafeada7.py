import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Momentum Component
    # Calculate daily price efficiency: (Close - Low) / (High - Low)
    price_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'])
    price_efficiency = price_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Compute 3-day average momentum
    momentum = price_efficiency.rolling(window=3, min_periods=1).mean()
    
    # Volume-Price Divergence Component
    # Compute 5-day volume trend slope
    volume_trend = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Compute 5-day price trend slope (using close prices)
    price_trend = df['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Calculate divergence as volume trend minus price trend
    divergence = volume_trend - price_trend
    
    # Signal Integration
    # Multiply momentum by divergence
    raw_signal = momentum * divergence
    
    # Apply hyperbolic tangent transformation
    factor = np.tanh(raw_signal)
    
    return factor
