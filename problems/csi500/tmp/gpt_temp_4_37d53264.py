import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Decay & Liquidity Absorption Alpha Factor
    Combines momentum exhaustion detection with liquidity absorption analysis
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Momentum Exhaustion Detection
    # Price Momentum Fade Rate
    momentum = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    # Momentum Reversal Thresholds (using rolling percentiles)
    momentum_ma = momentum.rolling(window=20, min_periods=10).mean()
    momentum_std = momentum.rolling(window=20, min_periods=10).std()
    momentum_saturation = (momentum - momentum_ma) / momentum_std
    
    # 2. Liquidity Absorption Analysis
    # Volume Absorption Efficiency
    price_change_abs = np.abs(data['close'] - data['close'].shift(1))
    volume_efficiency = data['volume'] / np.maximum(price_change_abs, 0.001)  # Avoid division by zero
    
    # Intraday Liquidity Asymmetry
    liquidity_asymmetry = (data['open'] * data['volume']) / (data['close'] * data['volume'])
    
    # 3. Price Efficiency Integration
    # Momentum-Adjusted Efficiency
    price_range = data['high'] - data['low']
    efficiency_ratio = price_range / np.maximum(price_change_abs, 0.001)
    
    # Liquidity-Momentum Alignment
    momentum_alignment = momentum_saturation * volume_efficiency.rolling(window=5, min_periods=3).mean()
    
    # 4. Composite Alpha Generation
    # Momentum Fade × Absorption Build-up
    momentum_fade_score = -momentum_saturation * efficiency_ratio
    absorption_build_up = volume_efficiency * liquidity_asymmetry
    
    # Multi-Timeframe Validation (5-day and 10-day momentum cycles)
    momentum_5d = momentum.rolling(window=5, min_periods=3).mean()
    momentum_10d = momentum.rolling(window=10, min_periods=5).mean()
    momentum_consistency = np.sign(momentum_5d) * np.sign(momentum_10d) * np.minimum(np.abs(momentum_5d), np.abs(momentum_10d))
    
    # Composite decay-absorption score
    composite_alpha = (
        momentum_fade_score * 0.4 +
        absorption_build_up * 0.3 +
        momentum_alignment * 0.2 +
        momentum_consistency * 0.1
    )
    
    # Cross-Sectional Ranking (z-score normalization)
    alpha_rank = composite_alpha.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    return alpha_rank
