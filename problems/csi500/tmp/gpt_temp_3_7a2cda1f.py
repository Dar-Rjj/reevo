import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Session Price-Volume Efficiency Divergence factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['price_range'] = data['high'] - data['low']
    
    # Intraday Efficiency Patterns
    # Opening Efficiency
    data['opening_gap'] = (data['open'] - data['prev_close']).fillna(0)
    data['opening_efficiency'] = data['opening_gap'] / (data['high'] - data['low']).replace(0, np.nan)
    data['opening_efficiency'] = data['opening_efficiency'].fillna(0)
    
    # Early momentum approximation (using daily data as proxy)
    data['early_momentum'] = (data['close'] - data['open']).fillna(0)
    
    # Mid-Day Efficiency
    data['mid_day_stability'] = data['price_range'].rolling(window=5, min_periods=1).mean()
    data['mid_day_efficiency_ratio'] = data['amount'] / (data['volume'] * data['price_range']).replace(0, np.nan)
    data['mid_day_efficiency_ratio'] = data['mid_day_efficiency_ratio'].fillna(0)
    
    # Closing Efficiency
    data['closing_momentum'] = (data['close'] - data['open']).fillna(0)  # Using open as proxy for 2h close
    data['closing_efficiency_ratio'] = data['amount'] / (data['volume'] * data['price_range']).replace(0, np.nan)
    data['closing_efficiency_ratio'] = data['closing_efficiency_ratio'].fillna(0)
    
    # Price-Level Efficiency Analysis
    # Approximation using daily ranges
    data['upper_range_volume'] = data['volume'] * 0.25  # Simplified approximation
    data['lower_range_volume'] = data['volume'] * 0.25  # Simplified approximation
    data['middle_range_volume'] = data['volume'] * 0.5   # Simplified approximation
    
    # Efficiency at different price levels (simplified)
    data['high_efficiency'] = data['amount'] / (data['upper_range_volume'] * data['price_range']).replace(0, np.nan)
    data['low_efficiency'] = data['amount'] / (data['lower_range_volume'] * data['price_range']).replace(0, np.nan)
    data['mid_efficiency'] = data['amount'] / (data['middle_range_volume'] * data['price_range']).replace(0, np.nan)
    
    data['high_efficiency'] = data['high_efficiency'].fillna(0)
    data['low_efficiency'] = data['low_efficiency'].fillna(0)
    data['mid_efficiency'] = data['mid_efficiency'].fillna(0)
    
    # Efficiency Momentum
    data['opening_to_mid_delta'] = data['opening_efficiency'] - data['mid_day_efficiency_ratio']
    data['mid_to_closing_delta'] = data['mid_day_efficiency_ratio'] - data['closing_efficiency_ratio']
    
    # Daily efficiency persistence
    data['efficiency_score'] = data['amount'] / (data['volume'] * data['price_range']).replace(0, np.nan)
    data['efficiency_score'] = data['efficiency_score'].fillna(0)
    
    # Consecutive high/low efficiency days
    data['high_efficiency_flag'] = (data['efficiency_score'] > data['efficiency_score'].rolling(window=20, min_periods=1).quantile(0.7)).astype(int)
    data['low_efficiency_flag'] = (data['efficiency_score'] < data['efficiency_score'].rolling(window=20, min_periods=1).quantile(0.3)).astype(int)
    
    data['consecutive_high'] = data['high_efficiency_flag'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    data['consecutive_low'] = data['low_efficiency_flag'].groupby(data.index).transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    
    # Efficiency trend
    data['efficiency_ma_3d'] = data['efficiency_score'].rolling(window=3, min_periods=1).mean()
    
    # Extreme Efficiency Events
    data['ultra_high_efficiency'] = (data['efficiency_score'] > data['efficiency_score'].rolling(window=20, min_periods=1).quantile(0.9)).astype(int)
    data['ultra_low_efficiency'] = (data['efficiency_score'] < data['efficiency_score'].rolling(window=20, min_periods=1).quantile(0.1)).astype(int)
    
    # Efficiency regime shifts
    data['efficiency_change'] = data['efficiency_score'].diff()
    data['sudden_improvement'] = (data['efficiency_change'] > data['efficiency_change'].rolling(window=20, min_periods=1).quantile(0.8)).astype(int)
    data['sudden_deterioration'] = (data['efficiency_change'] < data['efficiency_change'].rolling(window=20, min_periods=1).quantile(0.2)).astype(int)
    
    # Composite Alpha Generation
    # Cross-session divergence
    data['opening_closing_divergence'] = data['opening_efficiency'] - data['closing_efficiency_ratio']
    data['intraday_efficiency_trend'] = data['opening_to_mid_delta'] + data['mid_to_closing_delta']
    
    # Volume-efficiency integration
    data['volume_weighted_efficiency'] = data['efficiency_score'] * data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean()
    data['efficiency_adjusted_volume'] = data['volume'] / (1 + abs(data['efficiency_score']))
    
    # Multi-scale confirmation
    data['efficiency_momentum_alignment'] = (
        data['efficiency_ma_3d'].diff() * 
        data['efficiency_score'].diff()
    )
    
    # Extreme event filtering
    data['extreme_event_filter'] = 1 - (data['ultra_high_efficiency'] | data['ultra_low_efficiency'])
    
    # Final composite factor
    factor = (
        data['opening_closing_divergence'] * 0.15 +
        data['intraday_efficiency_trend'] * 0.12 +
        data['volume_weighted_efficiency'] * 0.18 +
        data['efficiency_adjusted_volume'] * 0.10 +
        data['efficiency_momentum_alignment'] * 0.15 +
        (data['high_efficiency'] - data['low_efficiency']) * 0.12 +
        data['consecutive_high'] * 0.08 -
        data['consecutive_low'] * 0.10
    ) * data['extreme_event_filter']
    
    # Clean and return
    factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
