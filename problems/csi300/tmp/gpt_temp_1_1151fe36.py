import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Short-term mean reversion: Deviation from 3D rolling mean scaled by 3D rolling std
    mean_reversion = (df['close'] - df['close'].rolling(3).mean()) / df['close'].rolling(3).std()
    
    # Long-term momentum: Log-transformed price ratio over 10D period
    log_price_ratio = np.log(df['close'] / df['close'].shift(10))
    
    # Volatility normalization: Average True Range (ATR) over 10D rolling window
    true_range = (df['high'] - df['low']).rolling(10).mean()
    
    # Volume z-score for liquidity confirmation
    volume_z_score = (df['volume'] - df['volume'].rolling(5).mean()) / df['volume'].rolling(5).std()
    
    # Blend momentum and mean reversion with explicit weights (0.7 momentum, 0.3 reversion)
    blended_factor = 0.7 * log_price_ratio + 0.3 * mean_reversion
    
    # Normalize blended factor by 10D volatility (ATR)
    risk_adjusted_factor = blended_factor / (true_range + 1e-7)
    
    # Apply volume z-score weighting
    final_factor = risk_adjusted_factor * volume_z_score
    
    # Cap extremes using rolling 5D percentiles
    factor_percentile = final_factor.rolling(5).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    final_factor = final_factor.clip(lower=factor_percentile.quantile(0.1), upper=factor_percentile.quantile(0.9))
    
    return final_factor
