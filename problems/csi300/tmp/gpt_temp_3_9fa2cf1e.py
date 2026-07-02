import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced momentum-reversal factor with dynamic regime adaptation, multi-timescale signals, and volume-driven intraday efficiency.
    Key improvements:
    1) Dynamic momentum-reversal weights based on volatility and volume regimes
    2) Multi-timescale momentum (1D, 3D, 5D) with adaptive volume confirmation
    3) Intraday efficiency weighted by combined volume and price trends
    4) Balanced reversal strength across volatility and volume regimes
    """
    # Volatility regime detection (ATR-based)
    atr = (df['high'] - df['low']).rolling(5).mean()
    high_vol = (atr > 1.2 * atr.rolling(20).mean()).astype(float)
    low_vol = 1 - high_vol
    
    # Volume regime detection (rolling percentile)
    vol_pct = df['volume'].rolling(20).apply(lambda x: x.rank(pct=True).iloc[-1])
    high_vol_regime = (vol_pct > 0.8).astype(float)
    low_vol_regime = 1 - high_vol_regime
    
    # Multi-timescale momentum signals
    mom_1d = df['close'].pct_change(1)
    mom_3d = df['close'].pct_change(3)
    mom_5d = df['close'].pct_change(5)
    momentum = (
        0.4 * mom_1d * low_vol * low_vol_regime + 
        0.3 * mom_3d * low_vol * high_vol_regime + 
        0.3 * mom_5d * high_vol * low_vol_regime
    )
    
    # Adaptive reversal signals (stronger in high volatility or volume regimes)
    reversal_short = -df['close'].pct_change(3)
    reversal_long = -df['close'].pct_change(5)
    reversal = (
        0.5 * reversal_short * high_vol * high_vol_regime + 
        0.3 * reversal_short * low_vol * high_vol_regime + 
        0.2 * reversal_long * high_vol * low_vol_regime
    )
    
    # Intraday efficiency (close-to-open vs range) weighted by volume and price trends
    overnight = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    daytime = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    intraday_eff = overnight - daytime
    vol_trend = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    price_trend = df['close'].rolling(5).mean() / df['close'].rolling(20).mean()
    eff_weight = 0.3 * vol_trend + 0.2 * price_trend
    
    alpha = (
        momentum * (1 - 0.5 * high_vol_regime) +  # Momentum stronger in low volume regimes
        reversal * (0.7 + 0.3 * high_vol_regime) * (1 - 0.4 * high_vol) +  # Reversal stronger in high volume or volatility
        intraday_eff * (0.15 + eff_weight)  # Intraday efficiency weighted by volume and price trends
    )
    
    return alpha
