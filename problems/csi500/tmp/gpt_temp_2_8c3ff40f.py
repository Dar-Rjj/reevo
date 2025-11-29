import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required technical indicators
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Calculate rolling statistics for various windows
    for window in [2, 3, 5, 10]:
        df[f'range_{window}d'] = df['true_range'].rolling(window=window).mean()
        df[f'volume_ma_{window}d'] = df['volume'].rolling(window=window).mean()
        df[f'amount_ma_{window}d'] = df['amount'].rolling(window=window).mean()
        df[f'close_ma_{window}d'] = df['close'].rolling(window=window).mean()
    
    # Component 1: Asymmetric Gap Reaction Efficiency
    df['gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['gap_abs'] = abs(df['gap'])
    
    # Gap filling measurement
    df['max_gap_recovery'] = np.where(
        df['gap'] > 0,
        (df['open'] - df['low']) / (df['open'] - df['prev_close']).replace(0, np.nan),
        (df['high'] - df['open']) / (df['prev_close'] - df['open']).replace(0, np.nan)
    )
    df['gap_persistence'] = np.where(
        df['gap_abs'] > 0,
        abs(df['close'] - df['prev_close']) / df['gap_abs'],
        0
    )
    
    gap_signal = np.where(
        (df['gap'] > 0) & (df['close'] > df['open']) & (df['gap_persistence'] > 0.7),
        1,
        np.where(
            (df['gap'] < 0) & (df['close'] < df['open']) & (df['gap_persistence'] > 0.7),
            -1,
            0
        )
    )
    
    # Component 2: Volume-Intensity Price Fracture
    df['price_change'] = df['close'].pct_change()
    df['volume_ratio'] = df['volume'] / df['volume_ma_3d']
    df['fracture_magnitude'] = abs(df['high'] - df['low']) / df['close_ma_3d']
    
    # Volume-price alignment
    df['vp_alignment'] = np.where(
        df['price_change'] * df['volume_ratio'] > 0,
        df['price_change'] * df['volume_ratio'],
        df['price_change'] * df['volume_ratio'] * -0.5
    )
    
    fracture_signal = df['vp_alignment'] * df['fracture_magnitude']
    
    # Component 3: Temporal Price Compression Elasticity
    df['range_ratio_3d'] = df['range_3d'] / df['range_10d']
    df['compression_elasticity'] = np.where(
        df['range_ratio_3d'] < 0.7,
        df['true_range'] / df['range_3d'],
        0
    )
    
    compression_signal = np.where(
        (df['compression_elasticity'] > 1.5) & (df['close'] > df['open']),
        df['compression_elasticity'],
        np.where(
            (df['compression_elasticity'] > 1.5) & (df['close'] < df['open']),
            -df['compression_elasticity'],
            0
        )
    )
    
    # Component 4: Differential Amount-Price Momentum
    df['price_momentum'] = df['close'].pct_change(3)
    df['amount_momentum'] = df['amount'].pct_change(3)
    df['price_accel'] = df['price_momentum'] - df['price_momentum'].shift(1)
    df['amount_accel'] = df['amount_momentum'] - df['amount_momentum'].shift(1)
    
    momentum_signal = np.where(
        (df['price_accel'] * df['amount_accel'] > 0) & (abs(df['price_accel']) > 0.01),
        df['price_accel'] * np.sign(df['amount_accel']),
        np.where(
            (df['price_accel'] * df['amount_accel'] < 0) & (abs(df['price_accel']) > 0.01),
            -df['price_accel'],
            0
        )
    )
    
    # Component 5: Intraday Structural Integrity
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    # Support/resistance tests
    support_test = np.where(
        (df['low'] <= df['prev_low']) & (df['close'] > df['prev_low']),
        1,  # Successful support test
        0
    )
    
    resistance_test = np.where(
        (df['high'] >= df['prev_high']) & (df['close'] < df['prev_high']),
        -1,  # Successful resistance test
        0
    )
    
    structure_signal = support_test + resistance_test
    
    # Component 6: Multi-Fractal Volatility Regime
    df['vol_short'] = df['true_range'].rolling(2).std()
    df['vol_medium'] = df['true_range'].rolling(5).std()
    df['vol_long'] = df['true_range'].rolling(10).std()
    
    df['vol_regime'] = np.where(
        df['vol_short'] > df['vol_medium'] * 1.2,
        1,  # High volatility regime
        np.where(
            df['vol_short'] < df['vol_medium'] * 0.8,
            -1,  # Low volatility regime
            0
        )
    )
    
    regime_signal = df['vol_regime'] * df['price_change']
    
    # Component 7: Price-Volume Temporal Decoupling
    df['price_lead'] = df['price_change'].shift(1)
    df['volume_lead'] = df['volume_ratio'].shift(1)
    
    decoupling_signal = np.where(
        (df['volume_lead'] > 1.5) & (df['price_change'] > 0.01),
        1,  # Volume leading price up
        np.where(
            (df['price_lead'] < -0.01) & (df['volume_ratio'] < 0.8),
            -1,  # Price leading volume down
            0
        )
    )
    
    # Combine all components with weights
    weights = [0.15, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10]
    components = [gap_signal, fracture_signal, compression_signal, 
                 momentum_signal, structure_signal, regime_signal, decoupling_signal]
    
    # Normalize each component
    normalized_components = []
    for comp in components:
        comp_series = pd.Series(comp, index=df.index)
        if comp_series.std() > 0:
            normalized_comp = (comp_series - comp_series.rolling(20).mean()) / comp_series.rolling(20).std()
        else:
            normalized_comp = comp_series * 0
        normalized_components.append(normalized_comp)
    
    # Weighted combination
    for i, date in enumerate(df.index):
        if i < 20:  # Skip initial period for reliable statistics
            result[date] = 0
            continue
            
        factor_value = 0
        for j, comp in enumerate(normalized_components):
            if not pd.isna(comp[date]):
                factor_value += weights[j] * comp[date]
        
        result[date] = factor_value
    
    # Final normalization
    result = (result - result.rolling(20).mean()) / result.rolling(20).std()
    
    return result.fillna(0)
