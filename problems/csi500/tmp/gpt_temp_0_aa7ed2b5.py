import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Gap-Fill Efficiency with Amount Fragmentation factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close', 'amount', 'volume']
    if not all(col in df.columns for col in required_cols):
        return result
    
    # Calculate basic metrics
    df = df.copy()
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_range'] = df['prev_high'] - df['prev_low']
    df['prev_range'] = df['prev_range'].replace(0, np.nan)
    
    # Gap Persistence and Fill Dynamics
    df['gap_size'] = (df['open'] - df['prev_close']) / df['prev_range']
    df['gap_fill_ratio'] = np.where(
        df['gap_size'] > 0,
        (df['high'] - df['prev_close']) / (df['open'] - df['prev_close']),
        (df['low'] - df['prev_close']) / (df['open'] - df['prev_close'])
    )
    df['gap_persistence'] = 1 - np.minimum(np.abs(df['gap_fill_ratio']), 1)
    
    # Amount Behavior During Gap Filling
    df['amount_intensity_ratio'] = df['amount'] / df['amount'].rolling(5, min_periods=1).mean()
    df['amount_skew'] = df['amount'].rolling(10, min_periods=3).apply(
        lambda x: (x - x.mean()).mean() / x.std() if x.std() > 0 else 0
    )
    
    # Amount-Gap Alignment
    df['amount_gap_alignment'] = np.where(
        df['gap_size'] * df['amount_intensity_ratio'] > 0,
        np.abs(df['gap_size']) * df['amount_intensity_ratio'],
        -np.abs(df['gap_size']) * df['amount_intensity_ratio']
    )
    
    # Range Efficiency with Flow Fragmentation
    df['daily_range'] = df['high'] - df['low']
    df['daily_range'] = df['daily_range'].replace(0, np.nan)
    df['range_efficiency'] = np.abs(df['close'] - df['open']) / df['daily_range']
    
    # Flow Fragmentation Dynamics
    df['amount_vol_ratio'] = df['amount'] / (df['volume'] + 1e-10)
    df['fragmentation_score'] = df['amount_vol_ratio'].rolling(5, min_periods=3).std()
    
    # Fragmentation-Range Interaction
    df['fragmentation_efficiency'] = df['range_efficiency'] / (1 + df['fragmentation_score'])
    
    # Volatility-Regime Gap Response
    df['volatility_5d'] = df['daily_range'].rolling(5, min_periods=3).std()
    df['volatility_regime'] = df['daily_range'] / (df['volatility_5d'] + 1e-10)
    
    # Gap Behavior in Volatility Regimes
    df['regime_gap_alignment'] = df['gap_persistence'] * df['volatility_regime']
    
    # Amount Support During Transitions
    df['amount_regime_support'] = df['amount_intensity_ratio'] * df['volatility_regime']
    
    # Regime-Gap-Amount Alignment
    df['regime_gap_amount_alignment'] = (
        df['regime_gap_alignment'] * df['amount_regime_support']
    )
    
    # Composite Factor Integration
    # Gap-Fill Efficiency Combination
    df['gap_fill_efficiency'] = (
        df['gap_persistence'] * df['amount_gap_alignment'] +
        df['range_efficiency'] * df['fragmentation_efficiency']
    ) / 2
    
    # Regime-Adaptive Weighting
    df['regime_aware_efficiency'] = (
        df['gap_fill_efficiency'] * df['regime_gap_amount_alignment']
    )
    
    # Session-Structure Enhancement
    df['intraday_momentum'] = (df['close'] - df['open']) / (df['daily_range'] + 1e-10)
    df['session_efficiency'] = df['range_efficiency'] * (1 + np.abs(df['intraday_momentum']))
    
    # Final Factor Construction
    df['composite_factor'] = (
        0.4 * df['regime_aware_efficiency'] +
        0.3 * df['gap_fill_efficiency'] +
        0.2 * df['session_efficiency'] +
        0.1 * df['fragmentation_efficiency']
    )
    
    # Cross-Sectional Normalization
    result = df['composite_factor']
    
    # Handle NaN values
    result = result.fillna(0)
    
    return result
