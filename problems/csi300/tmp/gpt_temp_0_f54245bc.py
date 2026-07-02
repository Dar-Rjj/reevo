import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor combining:
    1) Adaptive momentum-reversal blending using volatility regimes
    2) Robust volume confirmation with median-based spikes
    3) Intraday efficiency signals with VWAP gaps
    4) Clean price extremes detection with volume validation
    
    Improvements over v1:
    - Uses median-based volume spikes instead of exponential smoothing
    - Incorporates VWAP gap signals
    - Simpler extreme detection with tighter bands
    - More interpretable regime thresholds
    """
    # Adaptive volatility regimes using IQR-based thresholds
    true_range = df['high'] - df['low']
    vol_regime = true_range.rolling(20).apply(
        lambda x: 2 if x[-1] > x.quantile(0.75) else (0 if x[-1] < x.quantile(0.25) else 1)
    )
    
    # Median-based volume spikes (more robust than EMA)
    vol_median = df['volume'].rolling(20).median()
    vol_spike = (df['volume'] > 3.0 * vol_median).astype(float)
    
    # VWAP gap signal (intraday efficiency)
    vwap = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan).ffill()
    vwap_gap = (df['open'] - vwap.shift(1)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Clean price extremes with volume validation
    price_rank = df['close'].rolling(10).rank(pct=True)
    at_extreme = ((price_rank > 0.9) | (price_rank < 0.1)).astype(float)
    valid_extreme = at_extreme * vol_spike
    
    # Momentum/reversal blending by regime
    mom_short = df['close'].pct_change(3)
    mom_long = df['close'].pct_change(10)
    reversal = -df['close'].pct_change(5)
    
    # Regime-adaptive signal weights
    momentum = np.select(
        [vol_regime == 0, vol_regime == 1, vol_regime == 2],
        [0.7*mom_long + 0.3*mom_short, 0.5*mom_long + 0.5*mom_short, 0.3*mom_long + 0.7*mom_short]
    )
    
    reversal_strength = np.select(
        [vol_regime == 0, vol_regime == 1, vol_regime == 2],
        [0.3, 0.5, 0.7]
    )
    
    alpha = (
        momentum * (1 - valid_extreme) +  # Momentum dominates except at validated extremes
        reversal * reversal_strength * valid_extreme +  # Reversal triggers at extremes
        vwap_gap * 0.2  # Consistent small weight to VWAP gaps
    )
    
    return alpha
