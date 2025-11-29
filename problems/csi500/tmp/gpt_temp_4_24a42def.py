import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Cross-Sectional Momentum Asymmetry Alpha Factor
    """
    df = data.copy()
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Directional Momentum Imbalance
    # Asymmetric Return Patterns
    df['upward_intensity'] = (df['high'] - df['open']) / np.where(df['close'] - df['low'] != 0, df['close'] - df['low'], np.nan)
    df['downward_resistance'] = (df['open'] - df['low']) / np.where(df['high'] - df['close'] != 0, df['high'] - df['close'], np.nan)
    df['directional_bias'] = df['upward_intensity'] / df['downward_resistance']
    
    # Momentum Persistence Asymmetry
    df['bullish_persistence'] = np.where(df['close'] > df['open'], df['close'] - df['open'], 0)
    df['bearish_persistence'] = np.where(df['close'] < df['open'], df['open'] - df['close'], 0)
    df['persistence_asymmetry'] = df['bullish_persistence'] / np.where(df['bearish_persistence'] != 0, df['bearish_persistence'], np.nan)
    
    # Volume-Momentum Alignment
    df['bull_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['bear_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    df['volume_directional_bias'] = df['bull_volume'] / np.where(df['bear_volume'] != 0, df['bear_volume'], np.nan)
    
    # Price Path Efficiency
    # Intraday Path Optimization
    df['directness_ratio'] = (df['close'] - df['open']) / np.where(df['high'] - df['low'] != 0, df['high'] - df['low'], np.nan)
    df['path_curvature'] = (df['high'] + df['low']) / 2 - df['open']
    df['directness_3d'] = df['directness_ratio'].rolling(window=3, min_periods=1).mean()
    df['efficiency_momentum'] = df['directness_ratio'] / np.where(df['directness_3d'] != 0, df['directness_3d'], np.nan)
    
    # Resistance-Support Dynamics
    df['upper_resistance'] = (df['high'] - df['open']) / np.where(df['high'] - df['low'] != 0, df['high'] - df['low'], np.nan)
    df['lower_support'] = (df['open'] - df['low']) / np.where(df['high'] - df['low'] != 0, df['high'] - df['low'], np.nan)
    df['support_resistance_asymmetry'] = df['upper_resistance'] / np.where(df['lower_support'] != 0, df['lower_support'], np.nan)
    
    # Volume-Path Correlation
    df['volume_extremes'] = np.where((df['close'] > df['open']) | (df['close'] < df['open']), df['volume'], 0)
    df['path_volume_alignment'] = df['directness_ratio'] * df['volume_extremes']
    df['volume_change'] = df['volume'].pct_change()
    df['efficiency_change'] = df['directness_ratio'].pct_change()
    df['efficiency_volume_momentum'] = df['efficiency_change'] * df['volume_change']
    
    # Temporal Momentum Structure
    # Session Phase Momentum
    df['early_momentum'] = (df['high'] - df['open']) / np.where(df['open'] != 0, df['open'], np.nan)
    df['late_momentum'] = (df['close'] - df['low']) / np.where(df['low'] != 0, df['low'], np.nan)
    df['phase_momentum_divergence'] = df['early_momentum'] - df['late_momentum']
    
    # Momentum Acceleration Patterns
    df['early_momentum_3d'] = df['early_momentum'].rolling(window=3, min_periods=1).mean()
    df['late_momentum_3d'] = df['late_momentum'].rolling(window=3, min_periods=1).mean()
    df['opening_acceleration'] = df['early_momentum'] / np.where(df['early_momentum_3d'] != 0, df['early_momentum_3d'], np.nan)
    df['closing_acceleration'] = df['late_momentum'] / np.where(df['late_momentum_3d'] != 0, df['late_momentum_3d'], np.nan)
    df['acceleration_divergence'] = df['opening_acceleration'] - df['closing_acceleration']
    
    # Volume-Temporal Alignment
    df['total_volume_3d'] = df['volume'].rolling(window=3, min_periods=1).sum()
    df['early_volume_concentration'] = df['bull_volume'] / np.where(df['total_volume_3d'] != 0, df['total_volume_3d'], np.nan)
    df['temporal_volume_efficiency'] = df['phase_momentum_divergence'] * df['early_volume_concentration']
    
    # Gap Response Dynamics
    # Gap Absorption Efficiency
    df['prev_close'] = df['close'].shift(1)
    df['gap_magnitude'] = abs(df['open'] - df['prev_close']) / np.where(df['prev_close'] != 0, df['prev_close'], np.nan)
    df['gap_absorption'] = (df['close'] - df['open']) / np.where(df['gap_magnitude'] != 0, df['gap_magnitude'], np.nan)
    df['daily_range'] = df['high'] - df['low']
    df['absorption_efficiency'] = df['gap_absorption'] / np.where(df['daily_range'] != 0, df['daily_range'], np.nan)
    
    # Directional Gap Response
    df['positive_gap_response'] = np.where(df['open'] > df['prev_close'], df['close'] - df['open'], 0)
    df['negative_gap_response'] = np.where(df['open'] < df['prev_close'], df['open'] - df['close'], 0)
    df['response_asymmetry'] = df['positive_gap_response'] / np.where(df['negative_gap_response'] != 0, df['negative_gap_response'], np.nan)
    
    # Volume-Gap Interaction
    df['gap_volume_intensity'] = df['volume'] / np.where(df['gap_magnitude'] != 0, df['gap_magnitude'], np.nan)
    df['response_volume_confirmation'] = df['gap_absorption'] * df['gap_volume_intensity']
    
    # Range Expansion Asymmetry
    # Expansion Direction Bias
    df['upward_expansion'] = df['high'] - df['prev_close']
    df['downward_expansion'] = df['prev_close'] - df['low']
    df['expansion_bias'] = df['upward_expansion'] / np.where(df['downward_expansion'] != 0, df['downward_expansion'], np.nan)
    
    # Range Momentum Patterns
    df['current_range'] = df['high'] - df['low']
    df['range_3d_avg'] = df['current_range'].rolling(window=3, min_periods=1).mean()
    df['range_acceleration'] = df['current_range'] / np.where(df['range_3d_avg'] != 0, df['range_3d_avg'], np.nan)
    df['range_volume_correlation'] = df['current_range'].pct_change() * df['volume'].pct_change()
    
    # Breakout Asymmetry Detection
    df['upward_breakout'] = df['high'] - df['prev_close']
    df['downward_breakout'] = df['prev_close'] - df['low']
    df['breakout_asymmetry'] = df['upward_breakout'] / np.where(df['downward_breakout'] != 0, df['downward_breakout'], np.nan)
    
    # Final Alpha Factor Construction
    # Combine all asymmetry components with weights
    factors = [
        'directional_bias', 'persistence_asymmetry', 'volume_directional_bias',
        'efficiency_momentum', 'support_resistance_asymmetry', 'path_volume_alignment',
        'phase_momentum_divergence', 'acceleration_divergence', 'temporal_volume_efficiency',
        'absorption_efficiency', 'response_asymmetry', 'response_volume_confirmation',
        'expansion_bias', 'range_acceleration', 'breakout_asymmetry'
    ]
    
    # Calculate z-scores for cross-sectional ranking
    alpha_values = pd.Series(index=df.index, dtype=float)
    
    for date in df.index:
        day_data = df.loc[date]
        valid_factors = []
        
        for factor in factors:
            if factor in day_data and not np.isnan(day_data[factor]) and not np.isinf(day_data[factor]):
                valid_factors.append(day_data[factor])
        
        if len(valid_factors) > 0:
            # Simple average of normalized factor values
            alpha_values[date] = np.nanmean(valid_factors)
        else:
            alpha_values[date] = np.nan
    
    # Final smoothing and normalization
    alpha_values = alpha_values.rolling(window=3, min_periods=1).mean()
    
    return alpha_values
