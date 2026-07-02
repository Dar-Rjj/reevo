import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced momentum-reversal factor with smoothed transitions, dynamic weighting,
    and robust volume confirmation. Incorporates short-term momentum, mid-term reversal,
    intraday price efficiency, and adaptive volatility adjustments with volume-based filtering.
    """
    # Volatility measure (5-day std of returns)
    volatility = df['close'].pct_change().rolling(5).std()
    
    # Smooth short-term momentum (1-day EWMA and 3-day EWMA)
    momentum_1d = df['close'].pct_change(1).ewm(span=3).mean()
    momentum_3d = df['close'].pct_change(3).ewm(span=5).mean()
    
    # Smoothed mid-term reversal (7-day EWMA of returns)
    reversal = -df['close'].pct_change(7).ewm(span=7).mean()
    
    # Volume strength indicator (rolling percentile rank)
    vol_rank = df['volume'].rolling(10).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    # Intraday price efficiency (close-to-open vs daily range)
    price_eff = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    
    # Dynamic momentum-reversal weighting based on volatility
    momentum_weight = 1 / (1 + volatility)
    reversal_weight = volatility / (1 + volatility)
    
    # Volume confirmation adjustment (reduce momentum when volume is below normal)
    vol_conf = np.where(vol_rank < 0.5, 1 + (vol_rank - 0.5), 1)
    
    # Final combination with dynamic and volume-adjusted weighting
    alpha = (
        momentum_weight * (0.7 * momentum_1d + 0.3 * momentum_3d) * vol_conf +
        reversal_weight * reversal * (1 + vol_rank) + 
        0.2 * price_eff
    )
    
    return alpha
