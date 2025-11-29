import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Intraday Price-Volume Acceleration Divergence Factor
    """
    df = data.copy()
    
    # Define first and last hour (assuming 6.5 hour trading day)
    first_hour_end = 0.15  # First 15% of trading day
    last_hour_start = 0.85  # Last 15% of trading day
    
    # Calculate intraday price levels for first and last hour
    df['first_hour_high'] = df['high'].rolling(window=3, min_periods=1).apply(lambda x: x.iloc[:int(len(x)*first_hour_end)].max() if len(x) >= 3 else np.nan)
    df['first_hour_low'] = df['low'].rolling(window=3, min_periods=1).apply(lambda x: x.iloc[:int(len(x)*first_hour_end)].min() if len(x) >= 3 else np.nan)
    df['last_hour_high'] = df['high'].rolling(window=3, min_periods=1).apply(lambda x: x.iloc[int(len(x)*last_hour_start):].max() if len(x) >= 3 else np.nan)
    df['last_hour_low'] = df['low'].rolling(window=3, min_periods=1).apply(lambda x: x.iloc[int(len(x)*last_hour_start):].min() if len(x) >= 3 else np.nan)
    
    # Estimate first and last hour volume (proportional allocation)
    df['volume_first_hour'] = df['volume'] * first_hour_end
    df['volume_last_hour'] = df['volume'] * (1 - last_hour_start)
    
    # Early Session Acceleration Signal
    df['early_range_efficiency'] = (df['first_hour_high'] - df['open']) / (df['first_hour_high'] - df['first_hour_low'] + 1e-8)
    df['early_momentum_intensity'] = (df['first_hour_high'] - df['open']) * df['volume_first_hour']
    df['early_acceleration'] = df['early_range_efficiency'] * df['early_momentum_intensity']
    
    # Late Session Acceleration Signal
    df['late_recovery_efficiency'] = (df['close'] - df['last_hour_low']) / (df['last_hour_high'] - df['last_hour_low'] + 1e-8)
    df['late_momentum_intensity'] = (df['close'] - df['last_hour_low']) * df['volume_last_hour']
    df['late_acceleration'] = df['late_recovery_efficiency'] * df['late_momentum_intensity']
    
    # Session Acceleration Differential
    df['acceleration_gap'] = df['late_acceleration'] - df['early_acceleration']
    
    # Volume adjustment
    df['volume_15d_median'] = df['volume'].rolling(window=15, min_periods=5).median()
    df['volume_adjusted_gap'] = df['acceleration_gap'] * (df['volume'] / (df['volume_15d_median'] + 1e-8))
    
    # Volume Flow Acceleration
    df['early_volume_concentration'] = df['volume_first_hour'] / (df['volume'] + 1e-8)
    df['late_volume_concentration'] = df['volume_last_hour'] / (df['volume'] + 1e-8)
    df['volume_flow_acceleration'] = df['acceleration_gap'] * (df['late_volume_concentration'] / (df['early_volume_concentration'] + 1e-8))
    
    # Multi-Timeframe Pattern Reinforcement
    # Acceleration Pattern Persistence
    df['acceleration_direction'] = np.sign(df['acceleration_gap'])
    df['acceleration_streak'] = df['acceleration_direction'].groupby(df.index).expanding().apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1
    ).reset_index(level=0, drop=True)
    df['persistence_multiplier'] = 1 + (df['acceleration_streak'] - 1) * 0.1
    
    # Volume Pattern Confirmation
    df['volume_10d_mean'] = df['volume'].rolling(window=10, min_periods=5).mean()
    df['volume_trend_alignment'] = (df['volume'] > df['volume_10d_mean']).astype(int)
    
    df['amount_20d_median'] = df['amount'].rolling(window=20, min_periods=10).median()
    df['amount_flow_consistency'] = df['amount'] / (df['amount_20d_median'] + 1e-8)
    
    # Divergence Detection and Signal Enhancement
    # Price-Volume Divergence
    df['price_acceleration_5d_mean'] = df['acceleration_gap'].rolling(window=5, min_periods=3).mean()
    df['volume_acceleration_5d_mean'] = df['volume_flow_acceleration'].rolling(window=5, min_periods=3).mean()
    df['price_volume_divergence'] = df['acceleration_gap'] - df['volume_acceleration_5d_mean']
    
    # Session Divergence Strength
    df['acceleration_gap_5d_range'] = df['acceleration_gap'].rolling(window=5, min_periods=3).apply(lambda x: x.max() - x.min())
    df['divergence_magnitude'] = abs(df['acceleration_gap']) / (df['acceleration_gap_5d_range'] + 1e-8)
    
    # Market Microstructure Integration
    # Trading Quality Assessment
    df['price_efficiency'] = abs(df['close'] - df['open']) / ((df['high'] - df['low']) + 1e-8)
    df['efficiency_10d_avg'] = df['price_efficiency'].rolling(window=10, min_periods=5).mean()
    df['efficiency_context'] = df['price_efficiency'] / (df['efficiency_10d_avg'] + 1e-8)
    
    # Generate Final Alpha Factor
    # Combine components with appropriate weights
    df['base_signal'] = (
        0.4 * df['volume_adjusted_gap'] +
        0.3 * df['volume_flow_acceleration'] +
        0.3 * df['price_volume_divergence']
    )
    
    # Apply pattern persistence and volume confirmation
    df['enhanced_signal'] = (
        df['base_signal'] * 
        df['persistence_multiplier'] *
        (1 + 0.2 * df['volume_trend_alignment']) *
        (1 + 0.1 * np.log1p(df['amount_flow_consistency']))
    )
    
    # Weight by divergence detection strength
    df['divergence_weighted'] = df['enhanced_signal'] * (1 + 0.15 * df['divergence_magnitude'])
    
    # Adjust for market microstructure conditions
    df['final_factor'] = df['divergence_weighted'] * (0.8 + 0.2 * df['efficiency_context'])
    
    # Clean up and return
    result = df['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
