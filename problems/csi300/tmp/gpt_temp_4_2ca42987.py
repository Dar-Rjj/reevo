import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Initialize output Series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Price Impact component
    # delta close_t (1 period)
    delta_close = df['close'].diff(1)
    
    # rolling mean of volume (10 periods)
    rolling_volume = df['volume'].rolling(window=10, min_periods=1).mean()
    
    # ratio of absolute delta close to rolling volume
    price_impact = delta_close.abs() / rolling_volume
    
    # normalize with cross-sectional rank
    price_impact_norm = price_impact.rank(pct=True)
    
    # Order Flow Asymmetry component
    # EMA of high (span=5)
    ema_high = df['high'].ewm(span=5, adjust=False).mean()
    
    # EMA of low (span=5)
    ema_low = df['low'].ewm(span=5, adjust=False).mean()
    
    # difference between EMAs
    ema_diff = ema_high - ema_low
    
    # Volume Confirmation component
    # rolling mean of volume (20 periods, shifted by 10 to avoid lookahead)
    # Using expanding mean for first 10 periods to avoid NaN
    shifted_volume = df['volume'].shift(10)
    rolling_volume_20 = shifted_volume.expanding(min_periods=1).mean()
    for i in range(20, len(df)):
        rolling_volume_20.iloc[i] = shifted_volume.iloc[i-19:i+1].mean()
    
    # ratio of current volume to rolling mean
    volume_ratio = df['volume'] / rolling_volume_20
    
    # threshold comparison
    volume_confirmation = (volume_ratio > 1.5).astype(float)
    
    # Combine components
    factor = price_impact_norm + ema_diff + volume_confirmation
    
    return factor
