import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Morning Rejection-Momentum Signal
    data['retracement_ratio'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['morning_momentum'] = (data['high'] - data['open']) / (data['open'] + 1e-8)
    data['rejection_momentum'] = data['retracement_ratio'] * data['morning_momentum']
    
    # Afternoon Recovery-Momentum Signal
    data['recovery_ratio'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['afternoon_momentum'] = (data['close'] - data['low']) / (data['low'] + 1e-8)
    data['recovery_momentum'] = data['recovery_ratio'] * data['afternoon_momentum']
    
    # Momentum Acceleration with Volatility Context
    data['momentum_differential'] = data['afternoon_momentum'] - data['morning_momentum']
    data['close_median_20d'] = data['close'].rolling(window=20, min_periods=10).median()
    data['price_adjusted_acceleration'] = data['momentum_differential'] * (data['close'] / (data['close_median_20d'] + 1e-8))
    
    # Volatility-Enhanced Acceleration (assuming first and last hour data not available, using daily range as proxy)
    data['morning_volatility'] = (data['high'] - data['open']).rolling(window=5, min_periods=3).mean()
    data['afternoon_volatility'] = (data['close'] - data['low']).rolling(window=5, min_periods=3).mean()
    data['volatility_ratio'] = data['afternoon_volatility'] / (data['morning_volatility'] + 1e-8)
    data['volatility_enhanced_acceleration'] = data['price_adjusted_acceleration'] * data['volatility_ratio']
    
    # Pattern Persistence with Volume Confirmation
    data['rejection_momentum_3d_sum'] = data['rejection_momentum'].rolling(window=3, min_periods=2).sum()
    data['recovery_momentum_3d_sum'] = data['recovery_momentum'].rolling(window=3, min_periods=2).sum()
    
    # Track consecutive days with consistent patterns
    data['rejection_positive'] = (data['rejection_momentum'] > 0).astype(int)
    data['recovery_positive'] = (data['recovery_momentum'] > 0).astype(int)
    
    data['rejection_consecutive'] = data['rejection_positive'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    data['recovery_consecutive'] = data['recovery_positive'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    # Volume confirmation and weighting
    data['volume_median_10d'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['volume_confirmation'] = (data['volume'] > 1.2 * data['volume_median_10d']).astype(float)
    data['amount_mean_15d'] = data['amount'].rolling(window=15, min_periods=8).mean()
    data['amount_weighting'] = data['amount'] / (data['amount_mean_15d'] + 1e-8)
    
    data['persistence_weight'] = (data['rejection_consecutive'] + data['recovery_consecutive']) / 2
    data['volume_enhanced_persistence'] = data['persistence_weight'] * data['volume_confirmation'] * data['amount_weighting']
    
    # Divergence Detection
    data['rejection_momentum_5d_mean'] = data['rejection_momentum'].rolling(window=5, min_periods=3).mean()
    data['recovery_momentum_5d_mean'] = data['recovery_momentum'].rolling(window=5, min_periods=3).mean()
    
    data['rejection_divergence'] = data['rejection_momentum'] - data['rejection_momentum_5d_mean']
    data['recovery_divergence'] = data['recovery_momentum'] - data['recovery_momentum_5d_mean']
    
    data['acceleration_5d_mean'] = data['volatility_enhanced_acceleration'].rolling(window=5, min_periods=3).mean()
    data['acceleration_divergence'] = data['volatility_enhanced_acceleration'] - data['acceleration_5d_mean']
    
    # Pattern consistency divergence
    data['pattern_consistency'] = np.abs(data['rejection_momentum'] - data['recovery_momentum'])
    data['pattern_divergence_strength'] = data['pattern_consistency'] * (1 + np.abs(data['acceleration_divergence']))
    
    # Liquidity-Aware Signal Generation
    data['daily_range'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['range_5d_avg'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['liquidity_efficiency'] = data['daily_range'] / (data['range_5d_avg'] + 1e-8)
    
    # Final Alpha Factor Combination
    data['reversal_momentum_component'] = (data['rejection_momentum'] + data['recovery_momentum']) / 2
    data['acceleration_component'] = data['volatility_enhanced_acceleration']
    
    # Combine all components with appropriate weights
    data['alpha_factor'] = (
        data['reversal_momentum_component'] * 0.4 +
        data['acceleration_component'] * 0.3 +
        data['pattern_divergence_strength'] * 0.2 +
        data['volume_enhanced_persistence'] * 0.1
    ) * data['liquidity_efficiency']
    
    # Clean up and return
    alpha_series = data['alpha_factor'].copy()
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan)
    
    return alpha_series
