import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Calculate novel cross-sectional alpha factors using asymmetric gap momentum,
    volume-velocity divergence, price compression dynamics, and other microstructural patterns.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # 1. Asymmetric Gap Momentum
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Overnight gap persistence
    data['overnight_gap'] = (data['open'] - data['prev_close']) / (data['high'] - data['low'] + 1e-8)
    data['gap_filling_ratio'] = (data['close'] - data['open']) / (data['open'] - data['prev_close'] + 1e-8)
    
    # Gap momentum factor
    data['gap_momentum'] = np.where(
        (data['overnight_gap'].abs() > 0.01) & (data['gap_filling_ratio'].abs() < 0.3),
        data['overnight_gap'] * (1 - data['gap_filling_ratio'].abs()),
        0
    )
    
    # 2. Volume-Velocity Divergence
    data['price_velocity'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_acceleration'] = data['volume'] / data['volume_ma_5'] - 1
    
    # Velocity-volume divergence
    data['velocity_volume_div'] = np.where(
        data['price_velocity'].abs() > 0.1,
        data['price_velocity'] * np.sign(data['volume_acceleration']),
        0
    )
    
    # 3. Price Compression Dynamics
    data['daily_range'] = data['high'] - data['low']
    data['prev_range'] = data['daily_range'].shift(1)
    data['compression_ratio'] = data['daily_range'] / (data['prev_range'] + 1e-8)
    
    # Compression momentum
    data['compression_ma_3'] = data['compression_ratio'].rolling(window=3, min_periods=2).mean()
    data['compression_trend'] = data['compression_ratio'] - data['compression_ma_3']
    
    # 4. Microstructural Momentum Echo
    data['large_volume_threshold'] = data['volume'].rolling(window=20, min_periods=10).quantile(0.8)
    data['is_large_trade'] = data['volume'] > data['large_volume_threshold']
    data['price_move'] = data['close'] - data['open']
    
    # Echo strength (using rolling window to avoid lookahead)
    data['echo_strength'] = data['price_move'].rolling(window=3, min_periods=2).mean() / (data['volume'] + 1e-8)
    
    # 5. Temporal Price Distribution
    data['session_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['position_consistency'] = data['session_position'].rolling(window=5, min_periods=3).std()
    
    # Position trend
    data['position_trend'] = data['session_position'].rolling(window=5, min_periods=3).mean()
    
    # 6. Volume Barrier Penetration
    data['volume_weighted_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['vwap_volume'] = data['volume_weighted_price'] * data['volume']
    data['volume_concentration'] = data['vwap_volume'].rolling(window=10, min_periods=5).std()
    
    # 7. Momentum Fracture Detection
    data['momentum_1d'] = data['close'].pct_change(1)
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_gap'] = data['momentum_1d'] - data['momentum_3d'] / 3
    
    # Fracture signal
    data['momentum_fracture'] = np.where(
        data['momentum_gap'].abs() > data['momentum_1d'].abs(),
        data['momentum_gap'] * np.sign(data['volume_acceleration']),
        0
    )
    
    # 8. Price-Volume Synchronization
    data['price_move_abs'] = data['price_move'].abs()
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / (data['volume'].rolling(window=20, min_periods=10).std() + 1e-8)
    
    # Synchronization score
    data['synchronization'] = data['price_move_abs'] * data['volume_zscore'].abs() * np.sign(data['price_move'] * data['volume_zscore'])
    
    # Combine factors with weights
    weights = {
        'gap_momentum': 0.15,
        'velocity_volume_div': 0.15,
        'compression_trend': 0.12,
        'echo_strength': 0.12,
        'position_trend': 0.12,
        'volume_concentration': 0.12,
        'momentum_fracture': 0.12,
        'synchronization': 0.10
    }
    
    # Calculate weighted factor
    for date in data.index:
        if not data.loc[date].isna().any():
            factor_value = 0
            for factor, weight in weights.items():
                factor_value += data.loc[date, factor] * weight
            factor_values.loc[date] = factor_value
    
    # Fill NaN values with forward fill (using only past data)
    factor_values = factor_values.ffill()
    
    return factor_values
