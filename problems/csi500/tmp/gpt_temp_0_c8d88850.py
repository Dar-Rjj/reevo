import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Price Reversal Component
    reversal_ratio = (df['high'] - df['close']) / (df['close'] - df['low'] + 1e-8)
    reversal_signal = np.log1p(reversal_ratio)
    
    vol_regime = df['close'].rolling(window=10).std() / (df['close'].rolling(window=60).std() + 1e-8)
    vol_regime_reversal = reversal_signal * vol_regime
    
    # Volume Acceleration Component
    volume_breakout_ratio = df['volume'] / (df['volume'].rolling(window=20).max() + 1e-8)
    volume_acceleration = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-8)
    volume_signal = volume_breakout_ratio * volume_acceleration
    
    # Momentum Consistency Filter
    mom_3d = np.sign(df['close'] / df['close'].shift(3) - 1)
    mom_5d = np.sign(df['close'] / df['close'].shift(5) - 1)
    
    mom_consistency = pd.DataFrame({
        'mom_3d': mom_3d,
        'mom_5d': mom_5d
    })
    mom_consistency['consistent'] = (mom_consistency['mom_3d'] == mom_consistency['mom_5d']).astype(int)
    momentum_score = mom_consistency['consistent'].rolling(window=3, min_periods=1).mean()
    
    # Range Breakout Integration
    upper_breakout = (df['high'] - df['high'].rolling(window=10).max()) / (df['close'] + 1e-8)
    lower_breakout = (df['low'].rolling(window=10).min() - df['low']) / (df['close'] + 1e-8)
    
    breakout_signal = np.maximum(np.abs(upper_breakout), np.abs(lower_breakout))
    atr_scale = (df['high'] - df['low']).rolling(window=5).mean()
    range_breakout = breakout_signal * atr_scale
    
    # Composite Alpha Generation
    composite_alpha = vol_regime_reversal * volume_signal * momentum_score * range_breakout
    
    # Cross-sectional ranking
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    alpha_output = composite_alpha.groupby(composite_alpha.index).transform(cross_sectional_rank)
    
    return alpha_output
