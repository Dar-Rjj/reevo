import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Compression Detection
    tr_rolling_min = true_range.rolling(window=10, min_periods=5).min()
    tr_rolling_max = true_range.rolling(window=10, min_periods=5).max()
    
    # Compression ratio (how compressed current TR is relative to recent range)
    compression_ratio = (true_range - tr_rolling_min) / (tr_rolling_max - tr_rolling_min + 1e-8)
    
    # Intraday momentum and returns
    intraday_return = (df['close'] - df['open']) / df['open']
    five_day_momentum = df['close'] / df['close'].shift(5) - 1
    
    # Momentum divergence during compression
    compression_intensity = 1 - compression_ratio  # Higher when more compressed
    momentum_divergence = intraday_return * compression_intensity
    
    # Volume analysis
    volume_ma_20 = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = df['volume'] / (volume_ma_20 + 1e-8)
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    # Volume confirmation during compression
    volume_compression_confirmation = volume_ratio * compression_intensity
    
    # Breakout signal components
    # 1. Compression-to-expansion transition (TR expansion from compressed state)
    tr_expansion = true_range / (true_range.shift(1) + 1e-8) - 1
    compression_expansion_signal = tr_expansion * compression_intensity.shift(1)
    
    # 2. Momentum-Volume alignment
    momentum_volume_alignment = np.sign(intraday_return) * np.sign(volume_ratio - 1) * abs(intraday_return)
    
    # 3. Multi-timeframe momentum consistency
    one_day_momentum = df['close'] / df['close'].shift(1) - 1
    three_day_momentum = df['close'] / df['close'].shift(3) - 1
    
    momentum_consistency = (np.sign(one_day_momentum) + np.sign(three_day_momentum) + np.sign(five_day_momentum)) / 3
    
    # Combine all components for final factor
    breakout_signal = (
        compression_expansion_signal * 0.3 +
        momentum_divergence * 0.25 +
        volume_compression_confirmation * 0.2 +
        momentum_volume_alignment * 0.15 +
        momentum_consistency * 0.1
    )
    
    # Normalize the factor
    factor = breakout_signal.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8)
    )
    
    return factor
