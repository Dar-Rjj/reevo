import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Reversal Components
    # Intraday Return Reversal
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['intraday_reversal'] = -1 * data['intraday_return']
    
    # Overnight Gap Reversal
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    data['gap_reversal'] = -1 * data['overnight_gap']
    
    # Compute Volume Acceleration Patterns
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_ratio'] = data['volume'] / data['prev_volume']
    data['volume_acceleration'] = data['volume_ratio'] - 1
    
    # Volume-Price Divergence
    data['price_movement'] = np.sign(data['intraday_return'])
    data['volume_direction'] = np.sign(data['volume_acceleration'])
    data['volume_price_divergence'] = data['price_movement'] * data['volume_direction']
    
    # Assess Multi-timeframe Persistence
    # Short-term Momentum Consistency
    data['reversal_alignment'] = np.sign(data['intraday_reversal']) * np.sign(data['gap_reversal'])
    data['same_direction_reversal'] = (data['reversal_alignment'] > 0).astype(int)
    
    # Count consecutive same-direction reversal days
    data['consecutive_reversal'] = 0
    for i in range(1, len(data)):
        if data['same_direction_reversal'].iloc[i] == 1:
            data['consecutive_reversal'].iloc[i] = data['consecutive_reversal'].iloc[i-1] + 1
    
    # Volume Pattern Persistence
    data['volume_accel_direction'] = np.sign(data['volume_acceleration'])
    data['volume_trend_strength'] = 0
    
    # Track volume acceleration direction over 3 days
    for i in range(2, len(data)):
        current_dir = data['volume_accel_direction'].iloc[i]
        prev_dir = data['volume_accel_direction'].iloc[i-1]
        prev_prev_dir = data['volume_accel_direction'].iloc[i-2]
        
        if current_dir == prev_dir == prev_prev_dir:
            data['volume_trend_strength'].iloc[i] = 3
        elif current_dir == prev_dir:
            data['volume_trend_strength'].iloc[i] = 2
        else:
            data['volume_trend_strength'].iloc[i] = 1
    
    # Generate Reversal Strength Score
    # Combine Price Reversal Components
    data['weighted_reversal'] = (0.6 * data['intraday_reversal'] + 
                                0.4 * data['gap_reversal'])
    
    # Apply Volume Confirmation
    data['volume_confirmed_reversal'] = data['weighted_reversal'] * data['volume_acceleration']
    
    # Adjust for volume trend persistence
    data['trend_adjusted_reversal'] = data['volume_confirmed_reversal'] * (1 + 0.1 * data['volume_trend_strength'])
    
    # Apply consecutive reversal multiplier
    data['persistence_boost'] = 1 + 0.05 * np.minimum(data['consecutive_reversal'], 5)
    data['enhanced_reversal'] = data['trend_adjusted_reversal'] * data['persistence_boost']
    
    # Final Alpha Factor Construction
    # Multi-day Aggregation with exponential weighting
    weights = np.array([0.5, 0.3, 0.2])  # Recent days heavier
    data['alpha_factor'] = np.nan
    
    for i in range(2, len(data)):
        if i >= 2:
            recent_values = data['enhanced_reversal'].iloc[i-2:i+1].values
            if len(recent_values) == 3:
                data['alpha_factor'].iloc[i] = np.sum(weights * recent_values)
    
    # Signal Enhancement
    # Identify extreme reversal conditions
    data['extreme_reversal'] = (data['enhanced_reversal'].abs() > 
                               data['enhanced_reversal'].rolling(window=20, min_periods=10).quantile(0.8)).astype(int)
    
    # Filter for high-volume confirmation periods
    volume_threshold = data['volume'].rolling(window=20, min_periods=10).quantile(0.7)
    data['high_volume'] = (data['volume'] > volume_threshold).astype(int)
    
    # Generate final alpha factor with risk adjustment
    data['final_alpha'] = data['alpha_factor'] * data['extreme_reversal'] * data['high_volume']
    
    # Risk adjustment based on volatility
    vol_window = 20
    data['price_volatility'] = data['close'].pct_change().rolling(window=vol_window, min_periods=10).std()
    data['volatility_adjustment'] = 1 / (1 + data['price_volatility'])
    
    data['risk_adjusted_alpha'] = data['final_alpha'] * data['volatility_adjustment']
    
    # Return the final alpha factor series
    return data['risk_adjusted_alpha']
