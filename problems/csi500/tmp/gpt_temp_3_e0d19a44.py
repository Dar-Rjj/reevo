import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['close_change'] = data['close'] - data['prev_close']
    data['abs_close_change'] = np.abs(data['close_change'])
    data['daily_range'] = data['high'] - data['low']
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    # Multi-Timeframe Range Efficiency Analysis
    # Raw range utilization calculation
    data['range_utilization'] = data['abs_close_change'] * data['daily_range']
    
    # 2-day vs 5-day range efficiency comparison
    data['range_util_2d'] = data['range_utilization'].rolling(window=2).mean()
    data['range_util_5d'] = data['range_utilization'].rolling(window=5).mean()
    data['range_efficiency_ratio'] = data['range_util_2d'] / data['range_util_5d']
    
    # Range compression/expansion patterns
    data['range_util_momentum'] = data['range_utilization'].pct_change(periods=3)
    data['range_sequence'] = data['daily_range'].rolling(window=5).apply(
        lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0
    )
    
    # Volatility-Adjusted Momentum Acceleration
    # Gap momentum efficiency patterns
    data['gap'] = data['open'] - data['prev_close']
    data['gap_momentum_efficiency'] = data['gap'] * data['close_change']
    
    # Multi-period momentum acceleration
    data['return_5d'] = data['close'].pct_change(periods=5)
    data['return_10d'] = data['close'].pct_change(periods=10)
    data['momentum_acceleration'] = (data['return_5d'] - data['return_10d']) * data['true_range']
    
    data['gap_momentum_3d'] = data['gap_momentum_efficiency'].rolling(window=3).mean()
    data['gap_momentum_5d'] = data['gap_momentum_efficiency'].rolling(window=5).mean()
    data['gap_momentum_diff'] = data['gap_momentum_3d'] - data['gap_momentum_5d']
    
    # Raw momentum persistence assessment
    data['momentum_persistence'] = data['close_change'].rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x[i]) == np.sign(x[i-1])]) / 4
    )
    
    # Liquidity-Efficiency Divergence Detection
    # Amount momentum analysis
    data['amount_momentum'] = data['amount'].pct_change(periods=3)
    data['amount_trend'] = data['amount_momentum'].rolling(window=5).mean()
    
    # Volume efficiency trend confirmation
    data['volume_efficiency'] = data['amount'] / (data['daily_range'] + 1e-8)
    data['volume_efficiency_trend'] = data['volume_efficiency'].pct_change(periods=3)
    
    # Range-liquidity efficiency divergence
    data['range_util_vs_amount'] = data['range_utilization'].pct_change(periods=3) - data['amount_momentum']
    
    # Directional consistency assessment
    data['directional_consistency'] = (
        np.sign(data['range_utilization'].pct_change(periods=3)) * 
        np.sign(data['amount_momentum'])
    )
    
    # Signal Generation Logic
    # Positive: Range compression with accelerating momentum & confirming liquidity
    positive_signal_1 = (
        (data['range_sequence'] < data['range_sequence'].rolling(window=20).quantile(0.3)) &  # Range compression
        (data['momentum_acceleration'] > 0) &  # Accelerating momentum
        (data['directional_consistency'] > 0)   # Confirming liquidity
    )
    
    # Positive: Range expansion with volatility-adjusted momentum efficiency
    positive_signal_2 = (
        (data['range_sequence'] > data['range_sequence'].rolling(window=20).quantile(0.7)) &  # Range expansion
        (data['gap_momentum_efficiency'] > data['gap_momentum_efficiency'].rolling(window=20).quantile(0.6)) &  # Momentum efficiency
        (data['momentum_acceleration'] > 0)  # Volatility-adjusted acceleration
    )
    
    # Negative: Range compression with decelerating momentum despite liquidity
    negative_signal_1 = (
        (data['range_sequence'] < data['range_sequence'].rolling(window=20).quantile(0.3)) &  # Range compression
        (data['momentum_acceleration'] < 0) &  # Decelerating momentum
        (data['directional_consistency'] > 0)   # Despite liquidity
    )
    
    # Negative: Range expansion without momentum acceleration & contrary liquidity
    negative_signal_2 = (
        (data['range_sequence'] > data['range_sequence'].rolling(window=20).quantile(0.7)) &  # Range expansion
        (data['momentum_acceleration'] <= 0) &  # No momentum acceleration
        (data['directional_consistency'] < 0)   # Contrary liquidity
    )
    
    # Combine signals into final factor
    data['factor'] = 0.0
    data.loc[positive_signal_1, 'factor'] += 1.0
    data.loc[positive_signal_2, 'factor'] += 1.0
    data.loc[negative_signal_1, 'factor'] -= 1.0
    data.loc[negative_signal_2, 'factor'] -= 1.0
    
    # Add weighted components for continuous factor
    data['factor'] += (
        0.3 * data['range_efficiency_ratio'].fillna(0) +
        0.25 * data['momentum_acceleration'].fillna(0) +
        0.2 * data['gap_momentum_diff'].fillna(0) +
        0.15 * data['range_util_vs_amount'].fillna(0) +
        0.1 * data['directional_consistency'].fillna(0)
    )
    
    # Normalize the factor
    data['factor'] = (data['factor'] - data['factor'].rolling(window=63, min_periods=1).mean()) / data['factor'].rolling(window=63, min_periods=1).std()
    
    return data['factor']
