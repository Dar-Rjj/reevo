import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-sectional alpha factor combining multiple heuristics:
    - Price-based momentum factors
    - Volume-price interaction factors  
    - Range-based volatility factors
    - Multi-timeframe convergence factors
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        return result
    
    # Price-based Momentum Factors
    # Intraday Momentum Persistence
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    df['intraday_return_3d'] = df['intraday_return'].rolling(window=3, min_periods=3).sum()
    df['intraday_autocorr'] = df['intraday_return'].rolling(window=5, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 5 else np.nan, raw=False
    )
    momentum_persistence = df['intraday_return_3d'] * df['intraday_autocorr'].fillna(0)
    
    # Overnight Gap Reversal
    df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['intraday_return_current'] = (df['close'] - df['open']) / df['open']
    gap_reversal = -df['overnight_gap'] * df['intraday_return_current']
    
    # Volume-Price Interaction Factors
    # Volume-Weighted Price Acceleration
    df['price_change_5d'] = (df['close'] - df['close'].shift(4)) / df['close'].shift(4)
    df['volume_change_5d'] = (df['volume'] - df['volume'].shift(4)) / (df['volume'].shift(4) + 1e-8)
    df['price_direction'] = np.sign(df['price_change_5d'])
    df['volume_direction'] = np.sign(df['volume_change_5d'])
    directional_consistency = (df['price_direction'] == df['volume_direction']).astype(int)
    volume_acceleration = df['price_change_5d'] * df['volume_change_5d'] * directional_consistency
    
    # Low-Volume Breakout Confirmation
    df['volume_20d_avg'] = df['volume'].rolling(window=20, min_periods=20).mean()
    df['low_volume_indicator'] = (df['volume'] < df['volume_20d_avg']).astype(int)
    df['price_range_5d'] = df['high'].rolling(window=5, min_periods=5).max() - df['low'].rolling(window=5, min_periods=5).min()
    df['breakout_direction'] = np.sign(df['close'] - (df['high'].rolling(window=5, min_periods=5).max() + 
                                                     df['low'].rolling(window=5, min_periods=5).min()) / 2)
    low_volume_breakout = df['low_volume_indicator'] * df['breakout_direction']
    
    # Range-Based Volatility Factors
    # High-Low Range Expansion
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['true_range_5d_avg'] = df['true_range'].rolling(window=5, min_periods=5).mean()
    df['range_expansion'] = df['true_range'] / (df['true_range_5d_avg'] + 1e-8)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    range_expansion_factor = df['range_expansion'] * df['close_position']
    
    # Volatility Clustering Mean Reversion
    df['abs_return'] = abs(df['close'].pct_change())
    df['vol_cluster_3d'] = df['abs_return'].rolling(window=3, min_periods=3).std()
    df['vol_cluster_10d'] = df['abs_return'].rolling(window=10, min_periods=10).std()
    volatility_clustering = (df['vol_cluster_3d'] / (df['vol_cluster_10d'] + 1e-8)) * -df['intraday_return_current']
    
    # Multi-Timeframe Convergence
    # Short-Long Term Alignment
    df['trend_3d'] = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    df['trend_10d'] = (df['close'] - df['close'].shift(9)) / df['close'].shift(9)
    df['trend_convergence'] = np.sign(df['trend_3d']) * np.sign(df['trend_10d'])
    df['volume_confirmation'] = (df['volume'] > df['volume'].rolling(window=10, min_periods=10).mean()).astype(int)
    timeframe_convergence = df['trend_convergence'] * df['volume_confirmation']
    
    # Opening Gap Consistency
    df['gap_filling'] = abs(df['intraday_return_current']) / (abs(df['overnight_gap']) + 1e-8)
    df['gap_consistency_5d'] = df['gap_filling'].rolling(window=5, min_periods=5).std()
    gap_consistency = (1 / (df['gap_consistency_5d'] + 1e-8)) * df['intraday_return_current']
    
    # Combine all factors with equal weights
    factors = [
        momentum_persistence,
        gap_reversal,
        volume_acceleration,
        low_volume_breakout,
        range_expansion_factor,
        volatility_clustering,
        timeframe_convergence,
        gap_consistency
    ]
    
    # Standardize and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        factor_standardized = (factor - factor.rolling(window=20, min_periods=20).mean()) / (factor.rolling(window=20, min_periods=20).std() + 1e-8)
        combined_factor = combined_factor + factor_standardized.fillna(0)
    
    result = combined_factor / len(factors)
    
    return result
