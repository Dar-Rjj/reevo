import numpy as np
def heuristics_v2(df):
    # Momentum Component
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'])
    momentum_std = intraday_momentum.rolling(window=5).std()
    volatility_adjusted_momentum = intraday_momentum / momentum_std
    
    # Range Compression Signal
    daily_range = (df['high'] - df['low']) / df['open']
    sma_range = daily_range.rolling(window=5).mean()
    normalized_range = daily_range / sma_range
    
    # Volume Expansion Signal
    sma_volume = df['volume'].rolling(window=10).mean()
    volume_surge = df['volume'] / sma_volume
    volume_expansion = volume_surge.apply(lambda x: np.log(x) if x > 1 else x - 1)
    
    # Combined Signal
    combined_signal = (volatility_adjusted_momentum * normalized_range * volume_expansion)
    z_score = combined_signal.rolling(window=5).apply(lambda x: (x[-1] - x.mean()) / x.std())
    
    return z_score
