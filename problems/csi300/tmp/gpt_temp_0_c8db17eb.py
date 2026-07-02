import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor incorporating:
    1. Dynamic momentum/reversal switching based on volatility regimes
    2. Volume-price confirmation with adaptive thresholds
    3. Non-linear price efficiency metric
    4. Pure cross-sectional ranking with sector-relative adjustment
    5. Asymmetric response to up/down moves
    """
    
    # 1. Dynamic volatility regime detection
    true_range = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()),
                                   abs(df['low'] - df['close'].shift())))
    atr = true_range.rolling(14).mean()
    vol_regime = np.where(atr > atr.rolling(50).mean(), 1, -1)  # 1=high vol, -1=low vol
    
    # 2. Asymmetric momentum construction
    ret_5 = df['close'].pct_change(5)
    pos_mom = np.where(ret_5 > 0, ret_5, 0)
    neg_mom = np.where(ret_5 < 0, abs(ret_5), 0)
    
    # 3. Volume confirmation signal
    vol_ma_ratio = df['volume'] / df['volume'].rolling(20).mean()
    price_dir = np.sign(df['close'].pct_change(3))
    vol_confirmation = np.log1p(vol_ma_ratio) * price_dir
    
    # 4. Price efficiency metric (non-linear range utilization)
    daily_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    eff_score = daily_efficiency.rolling(5).apply(lambda x: np.prod(1 + x) - 1)
    
    # 5. Cross-sectional ranking with regime adaptation
    def cross_rank(series):
        return series.groupby(level=0).rank(pct=True)
    
    # Dynamic signal blending based on volatility regime
    mom_signal = cross_rank(
        np.where(vol_regime > 0,
                neg_mom * 0.7 + pos_mom * 0.3,  # Favor reversals in high vol
                pos_mom * 0.8 + neg_mom * 0.2)  # Favor momentum in low vol
    )
    
    vol_signal = cross_rank(np.tanh(vol_confirmation * 2))
    eff_signal = cross_rank(np.sqrt(abs(eff_score)) * np.sign(eff_score))
    
    # 6. Sector-relative adjustment
    sector_rank = df.groupby(level=0)['close'].transform(lambda x: x.rank(pct=True))
    
    # Final composition with interaction terms
    alpha = (
        0.5 * mom_signal +
        0.3 * vol_signal * sector_rank +
        0.2 * eff_signal * (1 - abs(mom_signal - 0.5))  # Magnifies efficiency at extremes
    )
    
    return alpha
