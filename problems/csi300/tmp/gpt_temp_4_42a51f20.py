import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced momentum-reversal factor with multi-timescale signals, dynamic weighting, and binary volume spikes.
    Key improvements:
    1) Multi-timescale momentum and reversal signals (1D, 3D, 5D, 10D).
    2) Binary volume spike confirmation across different timeframes.
    3) Dynamic weighting based on volatility regimes and momentum strength.
    4) Regime detection using rolling percentiles.
    5) Price efficiency metrics adjusted for intraday dynamics.
    """
    # Multi-timescale momentum signals
    mom_1d = df['close'].pct_change(1)
    mom_3d = df['close'].pct_change(3)
    mom_5d = df['close'].pct_change(5)
    mom_10d = df['close'].pct_change(10)
    
    # Regime detection using rolling percentiles
    volatility = (df['high'] - df['low']).rolling(5).mean()
    vol_pctile = volatility.rolling(20).apply(lambda x: x.rank(pct=True).iloc[-1])
    high_vol = (vol_pctile > 0.75).astype(float)
    
    # Binary volume spike confirmation (short-term vs long-term)
    vol_spike_short = (df['volume'] > 2.5 * df['volume'].rolling(5).median()).astype(float)
    vol_spike_long = (df['volume'] > 1.5 * df['volume'].rolling(20).median()).astype(float)
    
    # Balanced momentum and reversal roles
    momentum = 0.3 * mom_1d + 0.25 * mom_3d + 0.25 * mom_5d + 0.2 * mom_10d
    reversal = -0.4 * mom_5d * high_vol - 0.3 * mom_10d * (1 - high_vol)
    
    # Price efficiency metrics (overnight vs daytime move)
    overnight = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    daytime = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    efficiency = (overnight - daytime)
    
    # Dynamic weighting based on volatility regimes and momentum strength
    momentum_strength = momentum.abs().rolling(5).mean()
    momentum_weight = 0.5 - 0.2 * high_vol + 0.1 * (momentum_strength > 0.02).astype(float)
    reversal_weight = 0.3 + 0.2 * high_vol - 0.1 * (momentum_strength > 0.02).astype(float)
    
    alpha = (
        momentum_weight * momentum +
        reversal_weight * reversal * (1 - 0.5 * vol_spike_short - 0.3 * vol_spike_long) +
        0.2 * efficiency  # Fade inefficient intraday moves
    )
    
    return alpha
