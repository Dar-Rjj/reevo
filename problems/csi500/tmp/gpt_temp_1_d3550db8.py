import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Flow Fragmentation and Resistance Dynamics Factor
    Combines session segmentation, price-level resistance, temporal asymmetry, and volume-flow distribution patterns
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic intraday metrics
    data['dollar_volume'] = data['close'] * data['volume']
    data['price_range'] = data['high'] - data['low']
    data['returns'] = data['close'].pct_change()
    data['intraday_volatility'] = data['price_range'] / data['close']
    
    # Session segmentation (assuming 6.5 hour trading day)
    # Morning session: first 3.25 hours, Afternoon: last 3.25 hours
    # For daily data, we'll approximate using rolling windows
    
    # Morning vs Afternoon Flow Distribution (5-day rolling)
    data['morning_flow'] = data['dollar_volume'].rolling(window=5).apply(
        lambda x: x.iloc[:3].sum() if len(x) == 5 else np.nan, raw=False
    )
    data['afternoon_flow'] = data['dollar_volume'].rolling(window=5).apply(
        lambda x: x.iloc[3:].sum() if len(x) == 5 else np.nan, raw=False
    )
    data['flow_distribution_ratio'] = data['morning_flow'] / data['afternoon_flow']
    
    # Early vs Late Session Range Efficiency
    data['early_range_efficiency'] = data['returns'].rolling(window=3).apply(
        lambda x: x.iloc[0] / (abs(x.iloc[0]) + 1e-8) if len(x) == 3 else np.nan, raw=False
    )
    data['late_range_efficiency'] = data['returns'].rolling(window=3).apply(
        lambda x: x.iloc[2] / (abs(x.iloc[2]) + 1e-8) if len(x) == 3 else np.nan, raw=False
    )
    data['session_efficiency_divergence'] = data['early_range_efficiency'] - data['late_range_efficiency']
    
    # Flow Concentration Shift Detection
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['flow_concentration_shift'] = data['volume_concentration'].diff(3)
    
    # Hourly Flow Fragmentation (approximated with rolling windows)
    data['hourly_flow_variance'] = data['dollar_volume'].rolling(window=5).var()
    data['flow_direction_changes'] = data['returns'].rolling(window=3).apply(
        lambda x: sum((x.iloc[i] * x.iloc[i+1] < 0) for i in range(len(x)-1)) if len(x) == 3 else np.nan, raw=False
    )
    
    # Micro-Flow Reversal Detection
    data['small_trade_threshold'] = data['volume'].rolling(window=10).quantile(0.3)
    data['small_trade_flow'] = (data['volume'] < data['small_trade_threshold']).astype(int) * data['returns']
    data['micro_reversal_frequency'] = (data['small_trade_flow'].rolling(window=5).apply(
        lambda x: sum((x.iloc[i] * x.iloc[i+1] < 0) for i in range(len(x)-1)) if len(x) == 5 else np.nan, raw=False
    ))
    
    # Flow Interruption Quality
    data['flow_continuity'] = data['returns'].rolling(window=3).apply(
        lambda x: abs(x.iloc[1]) / (abs(x.iloc[0]) + abs(x.iloc[2]) + 1e-8) if len(x) == 3 else np.nan, raw=False
    )
    data['sudden_flow_change'] = abs(data['returns'].diff(2))
    
    # Price-Level Flow Resistance Dynamics
    data['round_number_distance'] = abs(data['close'] - (data['close'] // 1).astype(int))
    data['round_number_attraction'] = data['round_number_distance'].rolling(window=5).mean()
    
    # Recent High/Low Flow Patterns
    data['recent_high'] = data['high'].rolling(window=10).max()
    data['recent_low'] = data['low'].rolling(window=10).min()
    data['near_high_resistance'] = (data['close'] / data['recent_high'] - 1).abs()
    data['near_low_support'] = (data['close'] / data['recent_low'] - 1).abs()
    
    # Flow Accumulation at Resistance
    data['pre_breakout_volume'] = data['volume'].shift(1) / data['volume'].rolling(window=5).mean()
    data['resistance_test_quality'] = (data['high'] / data['recent_high'] - 1).abs() * data['volume']
    
    # Flow Exhaustion at Boundaries
    data['failed_break_attempt'] = ((data['high'] >= data['recent_high']) & 
                                   (data['close'] < data['recent_high'])).astype(int)
    data['boundary_flow_deceleration'] = data['volume'].diff(3) / data['volume'].rolling(window=5).mean()
    
    # Temporal Flow Asymmetry Patterns
    data['morning_afternoon_divergence'] = data['flow_distribution_ratio'] * data['session_efficiency_divergence']
    data['flow_intensity_asymmetry'] = data['morning_flow'] - data['afternoon_flow']
    
    # Flow Momentum Persistence
    data['multi_hour_alignment'] = data['returns'].rolling(window=3).apply(
        lambda x: sum(x > 0) if len(x) == 3 else np.nan, raw=False
    )
    data['flow_momentum_decay'] = data['returns'].rolling(window=3).apply(
        lambda x: x.iloc[2] - x.iloc[0] if len(x) == 3 else np.nan, raw=False
    )
    
    # Volume-Flow Distribution Quality
    data['large_trade_impact'] = (data['volume'] > data['volume'].rolling(window=10).quantile(0.7)).astype(int) * data['returns']
    data['flow_concentration_efficiency'] = data['large_trade_impact'].rolling(window=5).std()
    
    # Flow Distribution Variance
    data['intraday_flow_variance'] = data['dollar_volume'].rolling(window=5).var() / data['dollar_volume'].rolling(window=5).mean()
    data['flow_distribution_quality'] = 1 / (1 + data['intraday_flow_variance'])
    
    # Flow Fragmentation Impact
    data['fragmentation_intensity'] = (
        data['flow_direction_changes'] * 
        data['micro_reversal_frequency'] * 
        data['intraday_flow_variance']
    )
    
    # Combine factors with appropriate weights
    factor = (
        -0.15 * data['flow_distribution_ratio'].fillna(0) +
        0.12 * data['session_efficiency_divergence'].fillna(0) +
        -0.10 * data['flow_concentration_shift'].fillna(0) +
        0.08 * data['flow_direction_changes'].fillna(0) +
        -0.09 * data['micro_reversal_frequency'].fillna(0) +
        0.11 * data['flow_continuity'].fillna(0) +
        -0.07 * data['round_number_attraction'].fillna(0) +
        0.13 * data['near_high_resistance'].fillna(0) +
        -0.14 * data['pre_breakout_volume'].fillna(0) +
        0.10 * data['failed_break_attempt'].fillna(0) +
        0.06 * data['morning_afternoon_divergence'].fillna(0) +
        0.09 * data['multi_hour_alignment'].fillna(0) +
        -0.08 * data['flow_momentum_decay'].fillna(0) +
        0.12 * data['flow_concentration_efficiency'].fillna(0) +
        -0.11 * data['fragmentation_intensity'].fillna(0)
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20).mean()) / (factor.rolling(window=20).std() + 1e-8)
    
    return factor
