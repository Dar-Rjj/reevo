import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Basic price-volume calculations
    data['price_move'] = data['close'] - data['open']
    data['abs_price_move'] = np.abs(data['price_move'])
    data['daily_range'] = data['high'] - data['low']
    
    # Intraday Price-Volume Divergence
    data['price_move_volume_ratio'] = data['price_move'] / data['volume']
    data['abs_price_volume_efficiency'] = data['abs_price_move'] / data['volume']
    data['high_low_range_volume_efficiency'] = data['daily_range'] / data['volume']
    
    # Range-Volume Efficiency Analysis
    data['total_range_efficiency'] = data['daily_range'] / data['volume']
    data['upside_range_efficiency'] = (data['high'] - data['open']) / data['volume']
    data['downside_range_efficiency'] = (data['open'] - data['low']) / data['volume']
    data['range_efficiency_asymmetry'] = data['upside_range_efficiency'] / (data['downside_range_efficiency'] + 1e-8)
    
    # Volume-Price Momentum Dynamics
    data['volume_weighted_intraday_move'] = data['price_move'] * data['volume']
    
    # Calculate rolling averages for comparison
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['range_3d_avg'] = data['daily_range'].rolling(window=3, min_periods=1).mean()
    
    # Volume acceleration effect
    data['prev_volume'] = data['volume'].shift(1)
    data['volume_acceleration_effect'] = (data['volume'] / (data['prev_volume'] + 1e-8)) * data['price_move']
    
    # Direction-based volume pressure
    data['same_direction_strength'] = np.where(data['price_move'] > 0, data['price_move'] * data['volume'], 0)
    data['opposite_direction_pressure'] = np.where(data['price_move'] < 0, data['price_move'] * data['volume'], 0)
    data['correlation_reversal_signal'] = data['same_direction_strength'] - data['opposite_direction_pressure']
    
    # Volume-constrained range analysis
    data['high_volume_narrow_range'] = np.where(data['volume'] > data['volume_3d_avg'], 
                                               data['daily_range'] / data['volume'], np.nan)
    data['low_volume_wide_range'] = np.where(data['volume'] < data['volume_3d_avg'], 
                                            data['daily_range'] / data['volume'], np.nan)
    data['volume_range_dispersion'] = data['high_volume_narrow_range'].fillna(0) - data['low_volume_wide_range'].fillna(0)
    
    # Multi-Timeframe calculations
    data['efficiency_3d_avg'] = data['total_range_efficiency'].rolling(window=3, min_periods=1).mean()
    data['volume_pressure_3d_avg'] = data['volume_weighted_intraday_move'].rolling(window=3, min_periods=1).mean()
    
    data['efficiency_trend'] = data['total_range_efficiency'] - data['efficiency_3d_avg']
    data['volume_pressure_trend'] = data['volume_weighted_intraday_move'] - data['volume_pressure_3d_avg']
    data['divergence_momentum'] = data['efficiency_trend'] * data['volume_pressure_trend']
    
    # Cross-sectional ranking (within each day)
    def rank_within_day(series):
        return series.rank(pct=True)
    
    data['relative_efficiency_rank'] = data.groupby(data.index)['total_range_efficiency'].transform(rank_within_day)
    data['volume_pressure_rank'] = data.groupby(data.index)['volume_weighted_intraday_move'].transform(rank_within_day)
    data['combined_efficiency_rank'] = data['relative_efficiency_rank'] * data['volume_pressure_rank']
    
    # Composite factor construction
    data['intraday_divergence_factor'] = data['price_move_volume_ratio'] * data['range_efficiency_asymmetry']
    data['volume_momentum_factor'] = data['volume_acceleration_effect'] * data['correlation_reversal_signal']
    data['range_efficiency_factor'] = data['range_efficiency_asymmetry'] * data['volume_range_dispersion']
    data['multi_timeframe_factor'] = data['divergence_momentum'] * data['combined_efficiency_rank']
    
    # Final alpha factor
    data['final_divergence_alpha'] = (data['intraday_divergence_factor'] * 
                                    data['volume_momentum_factor'] * 
                                    data['range_efficiency_factor'] * 
                                    data['multi_timeframe_factor'])
    
    # Handle any remaining NaN values
    data['final_divergence_alpha'] = data['final_divergence_alpha'].fillna(0)
    
    return data['final_divergence_alpha']
