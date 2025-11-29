import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Transition Factor
    Captures multi-asymmetry volatility patterns, threshold break behavior, and cross-sectional volatility structure
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate daily returns and volatility measures
    returns = df['close'].pct_change()
    high_low_range = (df['high'] - df['low']) / df['close'].shift(1)
    open_close_range = abs(df['close'] - df['open']) / df['open']
    
    # Multi-Asymmetry Volatility Patterns
    # Up vs Down Volatility Persistence
    up_days = returns > 0
    down_days = returns < 0
    
    # Rolling volatility for up and down days separately
    up_vol = returns[up_days].rolling(window=10, min_periods=5).std()
    down_vol = returns[down_days].rolling(window=10, min_periods=5).std()
    
    # Ratio of consecutive up-day to down-day volatility
    vol_ratio = (up_vol / down_vol).fillna(1.0)
    
    # Duration of high-volatility clustering
    high_vol_threshold = returns.rolling(window=20).std().quantile(0.7)
    vol_clusters_up = (returns[up_days].rolling(window=5).std() > high_vol_threshold).rolling(window=5).sum()
    vol_clusters_down = (returns[down_days].rolling(window=5).std() > high_vol_threshold).rolling(window=5).sum()
    
    # Opening vs Closing Volatility Dynamics
    opening_vol = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['open'].rolling(window=5).mean()
    closing_vol = abs(df['close'] - df['open']) / (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min())
    
    # Volatility Threshold Break Behavior
    vol_breakout_threshold = returns.rolling(window=30).std().quantile(0.8)
    volatility_breakout = returns.rolling(window=5).std() > vol_breakout_threshold
    
    # Returns following volatility breakout
    post_break_returns = returns.shift(-1).rolling(window=3).mean()
    breakout_momentum = post_break_returns * volatility_breakout.shift(1)
    
    # Volume-Volatility Interaction
    volume_change = df['volume'].pct_change()
    vol_change = returns.rolling(window=5).std().pct_change()
    
    # Volume expansion during volatility regime changes
    vol_regime_change = abs(vol_change) > vol_change.rolling(window=20).std()
    volume_expansion = (volume_change > volume_change.rolling(window=20).quantile(0.7)) & vol_regime_change
    
    # Cross-Sectional Volatility Structure (using rolling percentiles as proxy)
    vol_percentile = returns.rolling(window=20).std().rolling(window=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 50 else np.nan
    )
    
    # Volatility regime consistency across timeframes
    short_term_vol = returns.rolling(window=5).std()
    medium_term_vol = returns.rolling(window=20).std()
    long_term_vol = returns.rolling(window=50).std()
    
    vol_consistency = (
        (short_term_vol / medium_term_vol).abs() + 
        (medium_term_vol / long_term_vol).abs() +
        (short_term_vol / long_term_vol).abs()
    )
    
    # Combine components with appropriate weights
    factor = (
        0.15 * vol_ratio.fillna(1.0) +
        0.12 * (vol_clusters_up.fillna(0) - vol_clusters_down.fillna(0)) +
        0.10 * (opening_vol - closing_vol).fillna(0) +
        0.18 * breakout_momentum.fillna(0) +
        0.15 * volume_expansion.fillna(0).astype(float) +
        0.15 * vol_percentile.fillna(0.5) +
        0.15 * (-vol_consistency.fillna(0))  # Negative for lower consistency being better
    )
    
    # Normalize and handle edge cases
    result = (factor - factor.rolling(window=50, min_periods=20).mean()) / factor.rolling(window=50, min_periods=20).std()
    result = result.fillna(0)
    
    return result
