import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Intraday Range Efficiency Momentum
    # Calculate True Range Efficiency: (Close - Open) / (High - Low)
    true_range_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    true_range_efficiency = true_range_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # 3-day cumulative efficiency persistence
    efficiency_persistence = true_range_efficiency.rolling(window=3, min_periods=1).sum()
    
    intraday_momentum = true_range_efficiency * efficiency_persistence
    
    # Volume-Price Divergence Acceleration
    # Price-Velocity Ratio: 2-day return / 5-day return
    price_2d_return = data['close'].pct_change(periods=2)
    price_5d_return = data['close'].pct_change(periods=5)
    price_velocity_ratio = price_2d_return / price_5d_return
    price_velocity_ratio = price_velocity_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Volume-Velocity Ratio: 2-day volume / 5-day volume
    volume_2d = data['volume'].rolling(window=2, min_periods=1).mean()
    volume_5d = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_velocity_ratio = volume_2d / volume_5d
    volume_velocity_ratio = volume_velocity_ratio.replace([np.inf, -np.inf], np.nan)
    
    volume_price_divergence = price_velocity_ratio * volume_velocity_ratio
    
    # Gap Absorption Strength
    # Measure Gap Absorption: (Close - Open) / (Open - Previous_Close)
    prev_close = data['close'].shift(1)
    gap_absorption = (data['close'] - data['open']) / (data['open'] - prev_close)
    gap_absorption = gap_absorption.replace([np.inf, -np.inf], np.nan)
    
    # Intraday recovery ratio: (High - Close) / (High - Low)
    intraday_recovery = (data['high'] - data['close']) / (data['high'] - data['low'])
    intraday_recovery = intraday_recovery.replace([np.inf, -np.inf], np.nan)
    
    gap_strength = gap_absorption * intraday_recovery
    
    # Volatility Compression Breakout
    # Volatility Compression: 5-day high-low range / 20-day high-low range
    range_5d = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    range_20d = (data['high'] - data['low']).rolling(window=20, min_periods=1).mean()
    volatility_compression = range_5d / range_20d
    volatility_compression = volatility_compression.replace([np.inf, -np.inf], np.nan)
    
    # Next-day price expansion: (High - Low) / Previous_Day_Range
    prev_day_range = (data['high'] - data['low']).shift(1)
    price_expansion = (data['high'] - data['low']) / prev_day_range
    price_expansion = price_expansion.replace([np.inf, -np.inf], np.nan)
    
    volatility_breakout = volatility_compression * price_expansion
    
    # Amount Flow Persistence
    # Amount Momentum: 3-day amount / 10-day amount
    amount_3d = data['amount'].rolling(window=3, min_periods=1).mean()
    amount_10d = data['amount'].rolling(window=10, min_periods=1).mean()
    amount_momentum = amount_3d / amount_10d
    amount_momentum = amount_momentum.replace([np.inf, -np.inf], np.nan)
    
    # Price-direction consistency over 5 days (sign consistency of returns)
    returns_5d = data['close'].pct_change(periods=5)
    price_consistency = returns_5d.rolling(window=5, min_periods=1).apply(
        lambda x: np.mean(np.sign(x)) if len(x) > 0 else 0, raw=True
    )
    
    amount_persistence = amount_momentum * price_consistency
    
    # Opening Range Capture Efficiency
    # Opening Capture: (High - Open) / (High - Low)
    opening_capture = (data['high'] - data['open']) / (data['high'] - data['low'])
    opening_capture = opening_capture.replace([np.inf, -np.inf], np.nan)
    
    # Closing strength: (Close - Low) / (High - Low)
    closing_strength = (data['close'] - data['low']) / (data['high'] - data['low'])
    closing_strength = closing_strength.replace([np.inf, -np.inf], np.nan)
    
    opening_efficiency = opening_capture * closing_strength
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'intraday_momentum': intraday_momentum,
        'volume_price_divergence': volume_price_divergence,
        'gap_strength': gap_strength,
        'volatility_breakout': volatility_breakout,
        'amount_persistence': amount_persistence,
        'opening_efficiency': opening_efficiency
    })
    
    # Z-score normalization for each factor (cross-sectional)
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    # Equal weighted combination
    factor = normalized_factors.mean(axis=1)
    
    return factor
