import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    An enhanced alpha factor combining:
    1. Volatility-normalized momentum (10-day price change adjusted by rolling volatility)
    2. Volume z-scores (volume relative to 20-day rolling mean and standard deviation)
    3. Cross-sectional median comparison (momentum relative to the median across all stocks)
    The factor avoids hardcoded weights and emphasizes interpretable components.
    """
    # Volatility-normalized momentum (10-day price change adjusted by rolling volatility)
    momentum_10d = df['close'].pct_change(10)
    volatility_10d = df['close'].pct_change().rolling(10).std()
    vol_normalized_momentum = momentum_10d / (volatility_10d + 1e-7)
    
    # Volume z-scores (volume relative to 20-day rolling mean and standard deviation)
    volume_mean_20d = df['volume'].rolling(20).mean()
    volume_std_20d = df['volume'].rolling(20).std()
    volume_zscore = (df['volume'] - volume_mean_20d) / (volume_std_20d + 1e-7)
    
    # Cross-sectional median comparison (momentum relative to the median across all stocks)
    cross_sectional_median = momentum_10d.groupby(df.index).transform(lambda x: x.median())
    median_adjusted_momentum = momentum_10d - cross_sectional_median
    
    # Combine components dynamically
    alpha_factor = vol_normalized_momentum * volume_zscore * median_adjusted_momentum
    
    return alpha_factor
