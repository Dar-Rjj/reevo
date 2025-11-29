import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Basic price and volume calculations
    data['price_move'] = data['close'] - data['open']
    data['daily_range'] = data['high'] - data['low']
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    
    # Intraday Price-Volume Divergence
    data['price_move_volume_ratio'] = data['price_move'] / data['volume']
    data['range_volume_efficiency'] = data['daily_range'] / data['volume']
    
    # Session-Based Divergence (simplified using daily data)
    # For first/last hour, we'll use the first/last 25% of daily range as proxy
    data['morning_efficiency'] = (data['daily_range'] * 0.25) / (data['volume'] * 0.25)
    data['afternoon_efficiency'] = (data['daily_range'] * 0.25) / (data['volume'] * 0.25)
    
    # Extreme Divergence Detection
    data['efficiency_acceleration'] = data['range_volume_efficiency'] - data['range_volume_efficiency'].shift(1)
    data['volume_pressure_indicator'] = data['price_move'] * data['volume']
    
    # Volume-Price Momentum Dynamics
    data['volume_weighted_intraday_move'] = data['price_move'] * data['volume']
    data['volume_acceleration_effect'] = data['volume_ratio'] * data['price_move']
    
    # Price-Volume Correlation Patterns
    data['same_direction_strength'] = np.where(data['price_move'] > 0, data['price_move'] * data['volume'], 0)
    data['opposite_direction_pressure'] = np.where(data['price_move'] < 0, data['price_move'] * data['volume'], 0)
    
    # Temporal Volume-Price Dynamics
    data['early_vs_late_momentum'] = data['morning_efficiency'] - data['afternoon_efficiency']
    
    # Volume spike detection
    volume_spike = data['volume'] > data['volume'].shift(1)
    data['volume_spike_price_response'] = np.where(volume_spike, data['price_move'] / data['volume'], 0)
    
    # Range-Volume Efficiency Analysis
    data['upside_range_efficiency'] = (data['high'] - data['open']) / data['volume']
    data['downside_range_efficiency'] = (data['open'] - data['low']) / data['volume']
    
    # False Range Volume Patterns
    high_break_fail = (data['high'] > data['high'].shift(1)) & (data['close'] < data['high'].shift(1))
    low_break_fail = (data['low'] < data['low'].shift(1)) & (data['close'] > data['low'].shift(1))
    
    data['failed_break_volume'] = np.where(high_break_fail, data['volume'], 0)
    data['failed_support_volume'] = np.where(low_break_fail, data['volume'], 0)
    
    # Volume-Constrained Range Analysis
    volume_avg = data['volume'].rolling(window=20, min_periods=1).mean()
    data['high_volume_narrow_range'] = np.where(data['volume'] > volume_avg, data['daily_range'] / data['volume'], 0)
    data['low_volume_wide_range'] = np.where(data['volume'] < volume_avg, data['daily_range'] / data['volume'], 0)
    
    # Multi-Timeframe Divergence Convergence
    data['efficiency_trend'] = data['range_volume_efficiency'] - data['range_volume_efficiency'].rolling(window=3, min_periods=1).mean()
    data['volume_pressure_trend'] = data['volume_pressure_indicator'] - data['volume_pressure_indicator'].rolling(window=3, min_periods=1).mean()
    
    # Efficiency Regime Detection
    efficiency_median = data['range_volume_efficiency'].rolling(window=20, min_periods=1).median()
    volume_median = data['volume'].rolling(window=20, min_periods=1).median()
    
    data['high_efficiency_low_volume'] = (data['range_volume_efficiency'] > efficiency_median) & (data['volume'] < volume_median)
    data['low_efficiency_high_volume'] = (data['range_volume_efficiency'] < efficiency_median) & (data['volume'] > volume_median)
    
    # Cross-Sectional Efficiency Ranking (daily cross-sectional rank)
    data['relative_efficiency_rank'] = data.groupby(data.index)['range_volume_efficiency'].transform(lambda x: x.rank(pct=True))
    data['volume_pressure_rank'] = data.groupby(data.index)['volume_weighted_intraday_move'].transform(lambda x: x.rank(pct=True))
    
    # Composite Divergence Alpha Construction
    data['intraday_divergence_factor'] = data['price_move_volume_ratio'] * (data['morning_efficiency'] / data['afternoon_efficiency'])
    data['volume_momentum_factor'] = data['volume_acceleration_effect'] * (data['same_direction_strength'] - data['opposite_direction_pressure'])
    data['range_efficiency_factor'] = (data['upside_range_efficiency'] - data['downside_range_efficiency']) * (data['high_volume_narrow_range'] - data['low_volume_wide_range'])
    data['multi_timeframe_factor'] = data['efficiency_trend'] * data['relative_efficiency_rank']
    
    # Final composite alpha factor
    alpha_factor = (
        data['intraday_divergence_factor'] * 0.25 +
        data['volume_momentum_factor'] * 0.25 +
        data['range_efficiency_factor'] * 0.25 +
        data['multi_timeframe_factor'] * 0.25
    )
    
    return alpha_factor
