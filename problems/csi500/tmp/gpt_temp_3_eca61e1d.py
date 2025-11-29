import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Decay with Volume Asymmetry
    # Calculate intraday momentum segments
    morning_momentum = (data['high'].rolling(window=120, min_periods=1).max() - data['open']) / data['open']
    afternoon_momentum = (data['close'] - data['low'].rolling(window=120, min_periods=1).min()) / data['low'].rolling(window=120, min_periods=1).min()
    full_day_momentum = (data['close'] - data['open']) / data['open']
    
    # Assess momentum decay patterns
    momentum_decay = morning_momentum.rolling(window=5).mean() - afternoon_momentum.rolling(window=5).mean()
    decay_acceleration = momentum_decay.diff(3)
    
    # Integrate volume asymmetry
    morning_volume = data['volume'].rolling(window=120, min_periods=1).mean()
    afternoon_volume = data['volume'].rolling(window=120, min_periods=1).mean()
    volume_asymmetry = (morning_volume - afternoon_volume) / (morning_volume + afternoon_volume + 1e-8)
    
    momentum_decay_factor = momentum_decay * volume_asymmetry * decay_acceleration
    
    # Range Compression Breakout with Volume Acceleration
    # Quantify trading range compression
    daily_range = (data['high'] - data['low']) / data['close']
    range_compression = daily_range.rolling(window=10).std() / daily_range.rolling(window=10).mean()
    
    # Detect breakout signals
    price_breakout = (data['close'] - data['open']) / data['open']
    volume_acceleration = data['volume'].pct_change(3)
    
    breakout_factor = range_compression * price_breakout * volume_acceleration
    
    # Volatility-Regime Momentum Reversal
    # Characterize volatility environment
    volatility = data['close'].pct_change().rolling(window=20).std()
    high_vol_regime = volatility > volatility.rolling(window=50).quantile(0.7)
    low_vol_regime = volatility < volatility.rolling(window=50).quantile(0.3)
    
    # Assess momentum reversal patterns
    momentum_reversal = -data['close'].pct_change(5)
    regime_reversal = momentum_reversal * high_vol_regime.astype(int) - momentum_reversal * low_vol_regime.astype(int)
    
    # Opening Imbalance with Trend Exhaustion
    # Analyze opening dynamics
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    opening_volume_ratio = data['volume'] / data['volume'].rolling(window=10).mean()
    opening_imbalance = opening_gap * opening_volume_ratio
    
    # Integrate trend exhaustion
    trend_strength = data['close'].rolling(window=10).apply(lambda x: (x[-1] - x[0]) / np.std(x) if np.std(x) > 0 else 0)
    volume_exhaustion = -data['volume'].pct_change(3)
    
    imbalance_factor = opening_imbalance * trend_strength * volume_exhaustion
    
    # Price Anchoring with Volume-Price Divergence
    # Establish dynamic price anchors
    vwap = (data['close'] * data['volume']).rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
    time_anchor = data['close'].rolling(window=10).mean()
    
    # Monitor divergence patterns
    price_position = (data['close'] - vwap) / vwap
    volume_divergence = (data['volume'] - data['volume'].rolling(window=10).mean()) / data['volume'].rolling(window=10).std()
    
    anchor_factor = price_position * volume_divergence
    
    # Composite factor combining all components
    composite_factor = (
        momentum_decay_factor.rank(pct=True) +
        breakout_factor.rank(pct=True) +
        regime_reversal.rank(pct=True) +
        imbalance_factor.rank(pct=True) +
        anchor_factor.rank(pct=True)
    )
    
    return composite_factor
