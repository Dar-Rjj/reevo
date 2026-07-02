import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Trend Acceleration Component
    # Second Derivative
    ema_15 = df['close'].ewm(span=15, adjust=False).mean()
    ema_25 = df['close'].ewm(span=25, adjust=False).mean()
    
    delta_ema_15 = ema_15.diff(periods=1)
    delta_ema_25 = ema_25.diff(periods=1)
    
    second_derivative = delta_ema_15 - delta_ema_25
    
    # Normalization
    rolling_std_close = df['close'].rolling(window=15).std()
    normalized_trend_acceleration = second_derivative / rolling_std_close
    
    # Liquidity Weighting
    # Volume-Adjusted
    rolling_mean_volume = df['volume'].rolling(window=25).mean()
    normalized_volume = df['volume'] / rolling_mean_volume
    
    volume_adjusted_component = normalized_volume * normalized_trend_acceleration
    
    # Spread Adjustment
    price_spread = df['high'] - df['low']
    spread_ratio = price_spread / df['close']
    volume_rank = df['volume'].rolling(window=15).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    spread_adjusted_component = volume_adjusted_component * spread_ratio * volume_rank
    
    # Decay Factor
    decay_factor = spread_adjusted_component.ewm(alpha=0.2, adjust=False).mean()
    
    # Liquidity-Weighted Trend Acceleration
    result = decay_factor
    
    return result
