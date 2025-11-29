import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Relative Strength & Volatility Transmission Factors
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Feature 1: Inter-Asset Momentum Spillover
    # Sector-relative momentum persistence using rolling correlation of returns
    returns = data['close'].pct_change()
    momentum_5d = returns.rolling(5).mean()
    momentum_10d = returns.rolling(10).mean()
    
    # Cross-momentum spillover: correlation between short and medium-term momentum
    momentum_spillover = momentum_5d.rolling(10).corr(momentum_10d)
    
    # Feature 2: Relative Volatility Regime Detection
    # High-vol vs low-vol behavior using rolling volatility ratios
    vol_5d = returns.rolling(5).std()
    vol_20d = returns.rolling(20).std()
    vol_regime = vol_5d / vol_20d
    
    # Volatility clustering detection using autocorrelation of squared returns
    squared_returns = returns ** 2
    vol_clustering = squared_returns.rolling(10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # Feature 3: Liquidity Gradient Analysis
    # Large vs small trade impact using volume-price relationship
    dollar_volume = data['close'] * data['volume']
    volume_5d_avg = data['volume'].rolling(5).mean()
    current_volume_ratio = data['volume'] / volume_5d_avg
    
    # Price impact measure: absolute returns normalized by volume
    price_impact = abs(returns) / (data['volume'].replace(0, np.nan))
    
    # Feature 4: Multi-Timeframe Convergence Signals
    # Alignment between different timeframe momentum signals
    momentum_alignment = (
        np.sign(momentum_5d.rolling(3).mean()) == 
        np.sign(momentum_10d.rolling(3).mean())
    ).astype(float)
    
    # Volatility regime confirmation across frequencies
    vol_short = returns.rolling(3).std()
    vol_medium = returns.rolling(10).std()
    vol_confirmation = (vol_short / vol_medium.rolling(3).mean()).fillna(0)
    
    # Combine factors with appropriate weights
    factor = (
        0.25 * momentum_spillover.fillna(0) +
        0.20 * vol_regime.fillna(0) +
        0.15 * vol_clustering.fillna(0) +
        0.15 * current_volume_ratio.fillna(0) +
        0.10 * (-price_impact.fillna(0)) +  # Negative weight as higher impact is worse
        0.10 * momentum_alignment.fillna(0) +
        0.05 * vol_confirmation.fillna(0)
    )
    
    # Final normalization
    factor = (factor - factor.rolling(20).mean().fillna(0)) / factor.rolling(20).std().replace(0, 1).fillna(1)
    
    return factor
