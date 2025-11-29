import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic daily metrics
    data['daily_range'] = data['high'] - data['low']
    data['daily_volume'] = data['volume']
    
    # Morning session (first hour) - using first hour data approximation
    # Assuming first hour data can be approximated from daily patterns
    data['morning_high'] = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x.max() if len(x) > 0 else np.nan)
    data['morning_low'] = data['low'].rolling(window=5, min_periods=1).apply(lambda x: x.min() if len(x) > 0 else np.nan)
    data['morning_close'] = data['close'].shift(1)  # Previous close as morning session close approximation
    
    # Morning range calculations
    data['morning_true_range'] = np.maximum(data['morning_high'], data['open']) - np.minimum(data['morning_low'], data['open'])
    data['morning_range_efficiency'] = (data['morning_close'] - data['open']) / data['morning_true_range']
    data['morning_range_efficiency'] = data['morning_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Morning volume concentration (approximated)
    data['morning_volume'] = data['volume'].rolling(window=5, min_periods=1).apply(lambda x: x.mean() if len(x) > 0 else np.nan)
    data['morning_volume_concentration'] = data['morning_volume'] / data['daily_volume']
    
    # Morning efficiency-weighted signals
    data['efficiency_weighted_morning_momentum'] = (data['morning_close'] - data['open']) * data['morning_range_efficiency']
    data['volume_efficiency_coordination'] = data['morning_range_efficiency'] * data['morning_volume_concentration']
    
    # Morning range position efficiency
    morning_range = data['morning_high'] - data['morning_low']
    data['morning_range_position'] = (data['morning_close'] - data['morning_low']) / morning_range
    data['morning_range_position'] = data['morning_range_position'].replace([np.inf, -np.inf], np.nan)
    data['morning_range_position_efficiency'] = data['morning_range_position'] * data['morning_range_efficiency']
    
    # Afternoon session calculations
    data['afternoon_true_range'] = np.maximum(data['high'], data['morning_close']) - np.minimum(data['low'], data['morning_close'])
    data['afternoon_range_efficiency'] = (data['close'] - data['morning_close']) / data['afternoon_true_range']
    data['afternoon_range_efficiency'] = data['afternoon_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Afternoon volume concentration (approximated)
    data['afternoon_volume'] = data['volume'] - data['morning_volume']
    data['afternoon_volume_concentration'] = data['afternoon_volume'] / data['daily_volume']
    
    # Afternoon range position
    daily_range = data['high'] - data['low']
    data['afternoon_range_position'] = (data['close'] - data['low']) / daily_range
    data['afternoon_range_position'] = data['afternoon_range_position'].replace([np.inf, -np.inf], np.nan)
    
    # Cross-session efficiency synchronization
    data['range_efficiency_divergence'] = data['morning_range_efficiency'] - data['afternoon_range_efficiency']
    data['volume_concentration_shift'] = data['morning_volume_concentration'] - data['afternoon_volume_concentration']
    data['efficiency_weighted_range_transfer'] = data['afternoon_range_efficiency'] * (data['morning_true_range'] / data['afternoon_true_range'])
    data['efficiency_weighted_range_transfer'] = data['efficiency_weighted_range_transfer'].replace([np.inf, -np.inf], np.nan)
    
    # Price-level range efficiency analysis
    # Upper range efficiency dynamics
    upper_range_threshold = data['high'] - 0.25 * daily_range
    data['upper_range_volume_ratio'] = data['volume'] * (data['close'] > upper_range_threshold) / data['daily_volume']
    
    # Lower range efficiency dynamics
    lower_range_threshold = data['low'] + 0.25 * daily_range
    data['lower_range_volume_ratio'] = data['volume'] * (data['close'] < lower_range_threshold) / data['daily_volume']
    
    # Mid-range efficiency
    mid_range_threshold_high = data['high'] - 0.25 * daily_range
    mid_range_threshold_low = data['low'] + 0.25 * daily_range
    data['mid_range_volume_ratio'] = data['volume'] * ((data['close'] <= mid_range_threshold_high) & (data['close'] >= mid_range_threshold_low)) / data['daily_volume']
    
    # Multi-session efficiency momentum
    data['morning_to_afternoon_efficiency_delta'] = data['afternoon_range_efficiency'] - data['morning_range_efficiency']
    data['efficiency_momentum_alignment'] = np.sign(data['morning_range_efficiency']) * np.sign(data['afternoon_range_efficiency'])
    
    # Range-context efficiency momentum
    morning_range_expansion = data['morning_true_range'] / data['morning_true_range'].shift(1)
    data['efficiency_in_expanding_ranges'] = data['morning_range_efficiency'] * morning_range_expansion
    range_ratio = data['afternoon_true_range'] / data['morning_true_range']
    data['efficiency_in_contracting_ranges'] = data['afternoon_range_efficiency'] * (1 - range_ratio)
    data['efficiency_in_contracting_ranges'] = data['efficiency_in_contracting_ranges'].replace([np.inf, -np.inf], np.nan)
    
    # Cross-day range-efficiency patterns
    data['range_efficiency_trend'] = data['morning_range_efficiency'] - data['morning_range_efficiency'].shift(1)
    
    # Extreme range-efficiency events
    morning_efficiency_90th = data['morning_range_efficiency'].rolling(window=20, min_periods=10).quantile(0.9)
    morning_efficiency_10th = data['morning_range_efficiency'].rolling(window=20, min_periods=10).quantile(0.1)
    
    data['ultra_high_morning_efficiency'] = (data['morning_range_efficiency'] > morning_efficiency_90th).astype(float)
    data['ultra_low_morning_efficiency'] = (data['morning_range_efficiency'] < morning_efficiency_10th).astype(float)
    
    # Composite alpha generation
    # Cross-session range-efficiency integration
    morning_component = (
        data['efficiency_weighted_morning_momentum'] + 
        data['volume_efficiency_coordination'] + 
        data['morning_range_position_efficiency']
    ) / 3
    
    afternoon_component = (
        data['afternoon_range_efficiency'] * (data['close'] - data['morning_close']) + 
        data['efficiency_weighted_range_transfer'] + 
        data['afternoon_range_efficiency'] * data['afternoon_range_position']
    ) / 3
    
    session_synchronization = (
        data['range_efficiency_divergence'] + 
        data['volume_concentration_shift'] + 
        data['efficiency_momentum_alignment']
    ) / 3
    
    # Price-level range-efficiency combination
    upper_range_signals = data['upper_range_volume_ratio'] * data['afternoon_range_efficiency']
    lower_range_signals = data['lower_range_volume_ratio'] * data['morning_range_efficiency']
    mid_range_signals = data['mid_range_volume_ratio'] * data['morning_range_efficiency']
    
    price_level_component = (upper_range_signals + lower_range_signals + mid_range_signals) / 3
    
    # Multi-scale range-efficiency confirmation
    intraday_momentum = (
        data['morning_to_afternoon_efficiency_delta'] + 
        data['efficiency_in_expanding_ranges'].fillna(0) + 
        data['efficiency_in_contracting_ranges'].fillna(0)
    ) / 3
    
    cross_day_patterns = (
        data['range_efficiency_trend'].fillna(0) + 
        data['morning_volume_concentration'] * data['morning_range_efficiency'] + 
        data['afternoon_volume_concentration'] * data['afternoon_range_efficiency']
    ) / 3
    
    extreme_events = (
        data['ultra_high_morning_efficiency'] * data['morning_range_efficiency'] - 
        data['ultra_low_morning_efficiency'] * data['morning_range_efficiency']
    )
    
    # Final alpha composition
    alpha = (
        0.3 * morning_component +
        0.3 * afternoon_component +
        0.15 * session_synchronization +
        0.1 * price_level_component +
        0.08 * intraday_momentum +
        0.05 * cross_day_patterns +
        0.02 * extreme_events
    )
    
    # Clean up and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = alpha.fillna(method='ffill')
    
    return alpha
