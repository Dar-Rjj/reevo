import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price and volume features
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate Fractal Dimension using Hurst exponent approximation
    def hurst_exponent(series, window=20):
        lags = range(2, min(window, len(series)))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Rolling fractal dimension calculation
    df['fractal_dim'] = df['close'].rolling(window=20).apply(
        lambda x: hurst_exponent(x) if len(x) == 20 else np.nan, raw=False
    )
    
    # Volume fractal using volume volatility pattern
    df['volume_fractal'] = df['volume'].rolling(window=10).std() / df['volume'].rolling(window=10).mean()
    
    # Range compression measure
    df['range_compression'] = df['high_low_range'].rolling(window=5).std()
    
    # Price Fractal Acceleration calculations
    df['price_accel_short'] = (df['fractal_dim'] - df['fractal_dim'].shift(1)) / \
                             (df['fractal_dim'].shift(2) - df['fractal_dim'].shift(3) + 1e-8)
    
    df['price_accel_medium'] = (df['fractal_dim'].shift(5) - df['fractal_dim'].shift(6)) / \
                              (df['fractal_dim'].shift(10) - df['fractal_dim'].shift(11) + 1e-8)
    
    df['price_accel_long'] = (df['fractal_dim'].shift(20) - df['fractal_dim'].shift(21)) / \
                            (df['fractal_dim'].shift(40) - df['fractal_dim'].shift(41) + 1e-8)
    
    # Volume Fractal Acceleration calculations
    df['volume_accel_short'] = (df['volume_fractal'] / (df['volume_fractal'].shift(1) + 1e-8)) / \
                              (df['volume_fractal'].shift(2) / (df['volume_fractal'].shift(3) + 1e-8))
    
    df['volume_accel_medium'] = (df['volume_fractal'].shift(5) / (df['volume_fractal'].shift(6) + 1e-8)) / \
                               (df['volume_fractal'].shift(10) / (df['volume_fractal'].shift(11) + 1e-8))
    
    df['volume_accel_long'] = (df['volume_fractal'].shift(20) / (df['volume_fractal'].shift(21) + 1e-8)) / \
                             (df['volume_fractal'].shift(40) / (df['volume_fractal'].shift(41) + 1e-8))
    
    # Range Fracture Acceleration
    df['range_accel'] = (df['range_compression'] / (df['range_compression'].shift(1) + 1e-8)) / \
                       (df['range_compression'].shift(2) / (df['range_compression'].shift(3) + 1e-8))
    
    # Detect Price-Volume Fractal Divergence
    df['divergence_short'] = np.where(
        (df['price_accel_short'] > 0) & (df['volume_accel_short'] < 0), 1,
        np.where((df['price_accel_short'] < 0) & (df['volume_accel_short'] > 0), -1, 0)
    )
    
    df['divergence_medium'] = np.where(
        (df['price_accel_medium'] > 0) & (df['volume_accel_medium'] < 0), 1,
        np.where((df['price_accel_medium'] < 0) & (df['volume_accel_medium'] > 0), -1, 0)
    )
    
    df['divergence_long'] = np.where(
        (df['price_accel_long'] > 0) & (df['volume_accel_long'] < 0), 1,
        np.where((df['price_accel_long'] < 0) & (df['volume_accel_long'] > 0), -1, 0)
    )
    
    # Multi-timeframe divergence persistence
    df['divergence_persistence'] = (
        df['divergence_short'].rolling(window=3).sum() * 0.4 +
        df['divergence_medium'].rolling(window=5).sum() * 0.35 +
        df['divergence_long'].rolling(window=8).sum() * 0.25
    )
    
    # Combine price and volume fractal acceleration signals
    price_signal = (
        df['price_accel_short'].fillna(0) * 0.5 +
        df['price_accel_medium'].fillna(0) * 0.3 +
        df['price_accel_long'].fillna(0) * 0.2
    )
    
    volume_signal = (
        df['volume_accel_short'].fillna(0) * 0.5 +
        df['volume_accel_medium'].fillna(0) * 0.3 +
        df['volume_accel_long'].fillna(0) * 0.2
    )
    
    # Generate final factor
    factor = (
        (price_signal - volume_signal) *  # Price-volume divergence
        df['range_accel'].fillna(1) *     # Scaled by range fracture
        df['divergence_persistence'].fillna(0)  # Weighted by persistence
    )
    
    return factor
