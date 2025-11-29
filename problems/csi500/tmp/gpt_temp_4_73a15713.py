import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price metrics
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    
    # Momentum Acceleration Measurement
    df['momentum_intensity'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['momentum_intensity'] = df['momentum_intensity'].fillna(0)
    
    # Historical momentum context
    df['momentum_5d_vol'] = df['momentum_intensity'].rolling(window=5, min_periods=3).std()
    df['momentum_stability'] = df['momentum_intensity'] / df['momentum_5d_vol'].replace(0, np.nan)
    df['momentum_stability'] = df['momentum_stability'].fillna(0)
    
    # Momentum regime shifts
    df['momentum_direction'] = np.sign(df['momentum_intensity'])
    df['momentum_dir_change'] = (df['momentum_direction'] != df['momentum_direction'].shift(1)).astype(int)
    df['consecutive_momentum_days'] = df.groupby((df['momentum_dir_change'] == 1).cumsum())['momentum_dir_change'].cumcount() + 1
    
    # Momentum range position
    df['momentum_20d_high'] = df['momentum_intensity'].rolling(window=20, min_periods=10).max()
    df['momentum_20d_low'] = df['momentum_intensity'].rolling(window=20, min_periods=10).min()
    df['momentum_range_position'] = (df['momentum_intensity'] - df['momentum_20d_low']) / (df['momentum_20d_high'] - df['momentum_20d_low']).replace(0, np.nan)
    df['momentum_range_position'] = df['momentum_range_position'].fillna(0.5)
    
    # Volume-Volatility Convergence Detection
    df['volume_5d_median'] = df['volume'].rolling(window=5, min_periods=3).median()
    df['volume_momentum'] = df['volume'] / df['volume_5d_median'].replace(0, np.nan)
    df['volume_momentum'] = df['volume_momentum'].fillna(1)
    
    df['true_range_5d_median'] = df['true_range'].rolling(window=5, min_periods=3).median()
    df['volatility_momentum'] = df['true_range'] / df['true_range_5d_median'].replace(0, np.nan)
    df['volatility_momentum'] = df['volatility_momentum'].fillna(1)
    
    df['convergence_magnitude'] = df['volume_momentum'] - df['volatility_momentum']
    
    # Structural Break Dynamics
    df['opening_structural_break'] = (df['high'] - df['low']) / df['prev_close'].replace(0, np.nan)
    df['opening_structural_break'] = df['opening_structural_break'].fillna(0)
    df['prev_structural_break'] = df['opening_structural_break'].shift(1)
    df['structural_break_intensity'] = df['opening_structural_break'] - df['prev_structural_break']
    
    # Volume acceleration analysis
    df['volume_2d_change'] = df['volume'].pct_change(periods=2)
    df['volume_structural_break'] = df['volume_2d_change'].abs()
    df['amount_per_trade'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['amount_per_trade'] = df['amount_per_trade'].fillna(0)
    
    # Gap Momentum Evaluation
    df['opening_momentum_gap'] = (df['open'] - df['prev_close']) / df['prev_close'].replace(0, np.nan)
    df['opening_momentum_gap'] = df['opening_momentum_gap'].fillna(0)
    df['momentum_gap_closure'] = (df['close'] - df['open']) / (df['open'] - df['prev_close']).replace(0, np.nan)
    df['momentum_gap_closure'] = df['momentum_gap_closure'].fillna(0)
    df['momentum_efficiency'] = df['momentum_gap_closure'].abs() / (df['volume'] + 1)
    
    # Regime Shift and Context Integration
    df['momentum_regime_intensity'] = df['consecutive_momentum_days'] * df['momentum_intensity'].abs()
    
    # Volatility level context
    df['volatility_15d_high'] = df['true_range'].rolling(window=15, min_periods=8).max()
    df['volatility_15d_low'] = df['true_range'].rolling(window=15, min_periods=8).min()
    df['volatility_position'] = (df['true_range'] - df['volatility_15d_low']) / (df['volatility_15d_high'] - df['volatility_15d_low']).replace(0, np.nan)
    df['volatility_position'] = df['volatility_position'].fillna(0.5)
    
    # Historical momentum direction bias
    df['positive_momentum_count'] = (df['momentum_intensity'] > 0).rolling(window=10, min_periods=5).sum()
    df['total_momentum_count'] = (df['momentum_intensity'].notna()).rolling(window=10, min_periods=5).sum()
    df['momentum_direction_bias'] = df['positive_momentum_count'] / df['total_momentum_count'].replace(0, np.nan)
    df['momentum_direction_bias'] = df['momentum_direction_bias'].fillna(0.5)
    
    # Volume-weighted momentum success
    df['volume_weighted_momentum'] = (df['momentum_intensity'] * df['volume']).rolling(window=10, min_periods=5).mean()
    df['momentum_success_rate'] = ((df['momentum_intensity'] * df['momentum_intensity'].shift(1)) > 0).rolling(window=10, min_periods=5).mean()
    
    # Core signal synthesis
    df['core_momentum_signal'] = (
        df['momentum_intensity'] * 0.3 +
        df['momentum_stability'] * 0.2 +
        df['convergence_magnitude'] * 0.25 +
        df['structural_break_intensity'] * 0.25
    )
    
    # Contextual enhancement
    df['contextual_enhancement'] = (
        df['momentum_direction_bias'] * 0.4 +
        df['volatility_position'] * 0.3 +
        df['opening_momentum_gap'] * 0.3
    )
    
    # Timing and persistence adjustments
    df['regime_persistence_multiplier'] = np.log1p(df['consecutive_momentum_days'])
    df['microstructure_acceleration'] = df['momentum_efficiency'] * df['amount_per_trade']
    
    # Final factor generation
    df['factor'] = (
        df['core_momentum_signal'] * 0.4 +
        df['contextual_enhancement'] * 0.3 +
        (df['regime_persistence_multiplier'] * df['core_momentum_signal']) * 0.15 +
        (df['microstructure_acceleration'] * df['momentum_success_rate']) * 0.15
    )
    
    # Normalize the factor
    df['factor_rank'] = df['factor'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    return df['factor_rank']
