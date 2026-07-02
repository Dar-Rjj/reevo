import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    # Calculate Intraday Return
    intraday_return = df['close'] / df['open'] - 1
    
    # Normalize by Volatility
    rolling_std_close = df['close'].pct_change().rolling(window=20).std()
    rolling_std_open = df['open'].pct_change().rolling(window=20).std()
    volatility = (rolling_std_close + rolling_std_open) / 2
    normalized_momentum = intraday_return / volatility.replace(0, np.nan)
    
    # Volume Confirmation Component
    # Current Volume vs Historical
    rolling_mean_volume = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / rolling_mean_volume.replace(0, np.nan)
    
    # Volume Trend Direction
    volume_slope = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(np.arange(5), x, 1)[0])
    volume_trend = volume_slope * normalized_momentum
    
    # Combine Components
    factor = normalized_momentum * volume_ratio * volume_trend
    return factor.dropna()
