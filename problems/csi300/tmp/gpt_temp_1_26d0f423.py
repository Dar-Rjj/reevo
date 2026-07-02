import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Short-term mean reversion: 5D price deviation from rolling mean
    mean_reversion = (df['close'] - df['close'].rolling(5).mean()) / df['close'].rolling(5).std()
    
    # Long-term momentum: 20D log price change for stability
    momentum = np.log(df['close'] / df['close'].shift(20))
    
    # Volatility normalization: 20D ATR for risk control
    atr = (df['high'] - df['low']).rolling(20).mean()
    
    # Blend short-term mean reversion (0.4) and long-term momentum (0.6)
    blended_factor = 0.4 * mean_reversion + 0.6 * momentum
    normalized_factor = blended_factor / (atr + 1e-7)
    
    # Volume weighting: 5D volume z-score for liquidity confirmation
    volume_z_score = (df['volume'] - df['volume'].rolling(5).mean()) / df['volume'].rolling(5).std()
    weighted_factor = normalized_factor * volume_z_score
    
    # Cap extremes using 10D rolling percentiles
    percentile_10 = weighted_factor.rolling(10).apply(lambda x: x.quantile(0.1))
    percentile_90 = weighted_factor.rolling(10).apply(lambda x: x.quantile(0.9))
    capped_factor = weighted_factor.clip(lower=percentile_10, upper=percentile_90)
    
    return capped_factor
