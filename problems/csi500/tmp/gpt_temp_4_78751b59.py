import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Sectional Range-Momentum Volatility Efficiency with Liquidity Dynamics
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Range calculations
    data['daily_range'] = data['high'] - data['low']
    data['prev_range'] = data['prev_high'] - data['prev_low']
    data['range_2day'] = (data['high'] - data['low']).rolling(window=2).mean()
    data['range_5day'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['range_utilization'] = abs(data['close'] - data['prev_close']) / (data['daily_range'] + 1e-8)
    
    # Multi-Timeframe Range Efficiency Patterns
    data['daily_range_efficiency'] = abs(data['close'] - data['prev_close']) * data['daily_range']
    data['range_compression'] = (data['daily_range'] < data['range_5day']).astype(int) * data['range_utilization']
    data['range_ratio_2_5'] = data['range_2day'] / (data['range_5day'] + 1e-8)
    
    # Range Sequence Analysis
    data['range_ratio_t_t1'] = data['daily_range'] / (data['prev_range'] + 1e-8)
    data['range_ratio_t1_t2'] = data['prev_range'] / (data['prev_high'].shift(1) - data['prev_low'].shift(1) + 1e-8)
    data['range_momentum'] = data['range_ratio_t_t1'] * data['range_utilization']
    data['range_util_5day'] = data['range_utilization'].rolling(window=5).mean()
    data['efficiency_momentum'] = data['range_utilization'] - data['range_util_5day']
    
    # Volatility-Adjusted Momentum Acceleration
    data['gap'] = data['open'] - data['prev_close']
    data['gap_momentum'] = data['gap'] * (data['close'] - data['prev_close'])
    data['vol_adjusted_gap'] = abs(data['gap']) / (data['daily_range'] + 1e-8) * (data['close'] - data['open'])
    data['gap_resolution_3day'] = (data['close'] - data['open']).rolling(window=3).mean()
    data['gap_momentum_3day'] = data['vol_adjusted_gap'] - data['gap_resolution_3day']
    
    # Momentum-Volatility Integration
    data['return_5day'] = data['close'] / data['close'].shift(5) - 1
    data['return_10day'] = data['close'] / data['close'].shift(10) - 1
    data['short_term_accel'] = data['return_5day'] * (data['daily_range'] / (data['prev_range'] + 1e-8))
    data['medium_term_accel'] = (data['return_5day'] - data['return_10day']) * data['daily_range']
    data['vol_ratio_momentum'] = data['short_term_accel'] * (data['daily_range'] / data['range_5day'])
    
    # Liquidity-Range Efficiency Divergence
    data['volume_range_efficiency'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['volume_range_3day'] = data['volume_range_efficiency'].rolling(window=3).mean()
    data['volume_range_divergence'] = data['volume_range_efficiency'] - data['volume_range_3day']
    
    # Amount-Range Divergence
    data['amount_3day'] = data['amount'].rolling(window=3).mean()
    data['amount_momentum'] = data['amount'] / (data['prev_amount'] + 1e-8)
    data['range_efficiency_amount'] = data['range_utilization'] * data['amount_momentum']
    data['liquidity_weighted_efficiency'] = (data['close'] - data['open']) * data['amount_momentum']
    
    # Composite Signal Generation
    # Positive Efficiency Signals
    data['signal_positive_1'] = data['range_compression'] * data['short_term_accel'] * data['volume_range_divergence']
    data['signal_positive_2'] = data['range_ratio_t_t1'] * data['vol_ratio_momentum']
    data['signal_positive_3'] = data['range_utilization'] * data['volume_range_divergence']
    
    # Negative Efficiency Signals
    data['signal_negative_1'] = data['range_compression'] * (-data['short_term_accel']) * (-data['volume_range_divergence'])
    data['signal_negative_2'] = data['range_ratio_t_t1'] * (-data['vol_ratio_momentum']) * (-data['volume_range_divergence'])
    data['signal_negative_3'] = (-data['range_utilization']) * (-data['volume_range_divergence'])
    
    # Final Composite Integration
    data['vol_momentum_core'] = data['range_utilization'] * data['short_term_accel']
    data['liquidity_enhanced'] = data['volume_range_divergence'] * data['amount_momentum']
    data['gap_resolution_integration'] = data['vol_adjusted_gap'] * data['vol_ratio_momentum']
    
    # Combine all components
    data['composite_efficiency'] = (
        data['vol_momentum_core'] + 
        data['liquidity_enhanced'] + 
        data['gap_resolution_integration'] +
        data['signal_positive_1'] + data['signal_positive_2'] + data['signal_positive_3'] -
        data['signal_negative_1'] - data['signal_negative_2'] - data['signal_negative_3']
    )
    
    # Factor Implementation Logic - Cross-sectional ranking
    # Calculate rolling z-score for cross-sectional normalization
    data['factor_value'] = data.groupby(data.index)['composite_efficiency'].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    
    return data['factor_value']
