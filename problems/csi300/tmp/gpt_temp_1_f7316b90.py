import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Price Trend Component
    price_change = (df['close'] - df['open']) / df['open']
    price_change_std = price_change.rolling(window=10, min_periods=1).std()
    normalized_price_change = price_change / price_change_std
    
    # Volume Trend Component
    volume_ma = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_change = df['volume'] / volume_ma
    volume_change_std = volume_change.rolling(window=10, min_periods=1).std()
    normalized_volume_change = (volume_change - 1) / volume_change_std
    
    # Divergence Score
    divergence_score = normalized_price_change * normalized_volume_change
    z_score = divergence_score.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return z_score
