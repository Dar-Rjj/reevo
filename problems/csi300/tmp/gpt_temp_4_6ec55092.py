import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced momentum-reversal factor with EWMA smoothing, dynamic timescales, 
    and robust volume-volatility alignment. Key improvements:
    1) Dual EWMA smoothing for momentum and reversal signals
    2) Dynamic volatility adaptation using percentile-based ATR
    3) Volume-spike confirmation integrated with volatility regimes
    4) Balanced intraday efficiency with overnight/daytime differentiation
    5) Optimized signal blending based on volatility and volume dynamics
    """
    # Dual EWMA-smoothed volatility regime detection
    atr = (df['high'] - df['low']).ewm(span=5, adjust=False).mean()
    atr_percentile = atr.rolling(20).apply(lambda x: x.rank(pct=True).iloc[-1])
    high_vol = (atr_percentile > 0.75).astype(float)
    low_vol = 1 - high_vol
    
    # Dual EWMA-smoothed momentum signals with dynamic weighting
    mom_short = df['close'].pct_change(1).ewm(span=3, adjust=False).mean()
    mom_mid = df['close'].pct_change(3).ewm(span=5, adjust=False).mean()
    mom_long = df['close'].pct_change(5).ewm(span=7, adjust=False).mean()
    momentum = (0.5 * mom_short * low_vol + 
               0.3 * mom_mid * low_vol + 
               0.2 * mom_long * high_vol)
    
    # Dual EWMA-smoothed reversal signals with volatility adaptation
    reversal_short = -df['close'].pct_change(3).ewm(span=5, adjust=False).mean()
    reversal_long = -df['close'].pct_change(5).ewm(span=7, adjust=False).mean()
    reversal = (0.7 * reversal_short * high_vol + 
                0.3 * reversal_long * low_vol)
    
    # Volume-spike confirmation with volatility regimes
    vol_spike = (df['volume'] > df['volume'].rolling(20).quantile(0.9)).astype(float)
    vol_weight = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
    
    # Intraday efficiency factor with overnight/daytime differentiation
    overnight_eff = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    daytime_eff = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    intraday_eff = overnight_eff - daytime_eff
    eff_weight = 0.4 * vol_weight  # Adjusted volume trend weight
    
    # Signal blending with volatility and volume dynamics
    alpha = (
        momentum * (1 - 0.3 * vol_spike) +  # Momentum dampened by spikes
        reversal * (0.8 + 0.2 * high_vol) * (1 - 0.2 * vol_spike) +  # Reversal adapted to spikes and volatility
        intraday_eff * (0.4 + eff_weight)  # Intraday efficiency with trend alignment
    )
    
    return alpha
