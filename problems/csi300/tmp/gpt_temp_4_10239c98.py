import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum Divergence components
    # EMA(close, window=10) rolling rank with window=20
    ema_10 = df['close'].ewm(span=10, adjust=False).mean()
    ema_10_rank = ema_10.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # EMA(close, window=5) rolling rank with window=10
    ema_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_5_rank = ema_5.rolling(window=10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Momentum Divergence (difference between the two ranks)
    momentum_divergence = ema_10_rank - ema_5_rank
    
    # Liquidity Adjustment components
    # Normalized Volume (current volume divided by 20-day rolling mean volume)
    volume_mean_20 = df['volume'].rolling(window=20).mean()
    normalized_volume = df['volume'] / volume_mean_20
    
    # Final factor: Momentum Divergence multiplied by Normalized Volume
    factor = momentum_divergence * normalized_volume
    
    return factor
