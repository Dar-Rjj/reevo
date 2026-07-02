import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # Calculate returns
    data['return'] = data['close'].pct_change()
    
    # Liquidity Ratio component
    # Calculate rolling mean volume over 20 days (using only past data)
    data['rolling_vol_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    # Calculate ratio of current volume to rolling mean
    data['volume_ratio'] = data['volume'] / data['rolling_vol_20']
    # Z-score the ratio cross-sectionally (daily)
    data['liquidity_zscore'] = data.groupby(data.index)['volume_ratio'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    # Volume-Weighted Momentum component
    # Calculate EMA of returns with span=10
    data['ema_return_10'] = data['return'].ewm(span=10, adjust=False).mean()
    # Normalize volume (current volume divided by mean volume over last 20 days)
    data['norm_volume'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    # Multiply EMA returns by normalized volume
    data['vol_weighted_momentum'] = data['ema_return_10'] * data['norm_volume']
    # Calculate rolling std of returns over 10 days for volatility scaling
    data['volatility'] = data['return'].rolling(window=10, min_periods=5).std()
    # Apply volatility scaling with baseline of 0.2
    data['scaled_momentum'] = data['vol_weighted_momentum'] / (data['volatility'] + 0.2)
    
    # Combine components with equal weight
    data['factor'] = 0.5 * data['liquidity_zscore'] + 0.5 * data['scaled_momentum']
    
    return data['factor']
