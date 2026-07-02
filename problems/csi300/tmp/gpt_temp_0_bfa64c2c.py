import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum component: Price change based on close-to-close returns (20D momentum)
    momentum = df['close'].pct_change(20)
    
    # Mean reversion component: Deviation from rolling mean scaled by rolling std (5D mean reversion)
    mean_reversion = (df['close'] - df['close'].rolling(5).mean()) / df['close'].rolling(5).std()
    
    # Blending short-term mean reversion with long-term momentum
    blended_factor = 0.6 * momentum - 0.4 * mean_reversion
    
    # Volatility normalization: Average true range over rolling window (20D ATR)
    true_range = (df['high'] - df['low']).rolling(20).mean()
    
    # Normalize by volatility (ATR)
    volatility_normalized_factor = blended_factor / (true_range + 1e-7)
    
    # Log-transform price ratios for stability (log of close-to-open ratio)
    price_ratio = np.log(df['close'] / df['open'])
    
    # Incorporate log-transformed price ratios
    price_stabilized_factor = volatility_normalized_factor * price_ratio
    
    # Volume weighting: Normalized volume trend using log transforms (5D volume z-score)
    volume_weight = np.log1p(df['volume']) - np.log1p(df['volume'].rolling(5).mean())
    
    # Weight signals by volume z-score for liquidity confirmation
    liquidity_confirmed_factor = price_stabilized_factor * volume_weight
    
    # Cap extremes using rolling percentiles (10D)
    percentile_capped_factor = liquidity_confirmed_factor.rolling(10).apply(lambda x: np.percentile(x, 10))
    
    return percentile_capped_factor
