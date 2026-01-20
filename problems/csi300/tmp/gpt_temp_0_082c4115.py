import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Price Change Ratio
    df['price_change_ratio'] = (df['close'] - df['open']) / df['open']
    
    # Calculate 10-day EMA of Price Change Ratio
    df['ema_price_change_ratio'] = df['price_change_ratio'].ewm(span=10, adjust=False).mean()
    
    # Calculate Normalized Momentum
    df['normalized_momentum'] = df['price_change_ratio'] / df['ema_price_change_ratio']
    
    # Calculate 5-day SMA Volume
    df['sma_volume_5'] = df['volume'].rolling(window=5).mean()
    
    # Calculate Raw Volume Surge
    df['raw_volume_surge'] = df['volume'] / df['sma_volume_5']
    
    # Apply Symmetric Transform
    df['symmetric_volume_surge'] = np.where(
        df['raw_volume_surge'] > 1,
        np.sqrt(df['raw_volume_surge']),
        df['raw_volume_surge']**2
    )
    
    # Combine Price Momentum and Volume Surge
    df['combined_factor'] = df['normalized_momentum'] * df['symmetric_volume_surge']
    
    # Apply 3-day Rolling Z-Score
    df['rolling_z_score'] = df['combined_factor'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std(), raw=False
    )
    
    return df['rolling_z_score']
