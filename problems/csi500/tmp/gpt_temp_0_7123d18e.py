import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Weighted Opening Momentum Efficiency (VWOME) factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price ratios and shifts
    data['prev_close'] = data['close'].shift(1)
    data['prev_close_2'] = data['close'].shift(2)
    data['prev_open'] = data['open'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_volume_2'] = data['volume'].shift(2)
    
    # 1. Opening Momentum Quality Assessment
    # Raw Opening Momentum components
    data['abs_gap_strength'] = np.abs(data['open'] - data['prev_close']) / data['prev_close']
    data['directional_gap_persistence'] = np.sign(data['open'] - data['prev_close']) * np.sign(data['prev_close'] - data['prev_close_2'])
    data['gap_acceleration'] = ((data['open'] - data['prev_close']) / data['prev_close'] - 
                               (data['prev_open'] - data['prev_close_2']) / data['prev_close_2'])
    
    # Intraday Momentum Efficiency
    high_low_range = data['high'] - data['low']
    high_low_range = np.where(high_low_range == 0, 1e-10, high_low_range)  # Avoid division by zero
    
    # Opening efficiency for positive gaps
    positive_gap_mask = (data['open'] > data['prev_close'])
    data['opening_efficiency'] = np.where(positive_gap_mask, 
                                         (data['high'] - data['open']) / high_low_range, 0)
    
    # Opening defense for negative gaps
    negative_gap_mask = (data['open'] < data['prev_close'])
    data['opening_defense'] = np.where(negative_gap_mask, 
                                      (data['open'] - data['low']) / high_low_range, 0)
    
    # Momentum sustainability
    data['momentum_sustainability'] = ((data['close'] - data['open']) / high_low_range * 
                                      np.sign(data['open'] - data['prev_close']))
    
    # 2. Volatility Context Integration
    # Relative Volatility Positioning
    data['daily_range'] = data['high'] - data['low']
    data['volatility_percentile'] = data['daily_range'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x) >= 10 else 0.5
    )
    
    # Volatility trend (5-day comparison)
    data['prev_range_5d'] = (data['high'] - data['low']).shift(5)
    data['volatility_trend'] = (data['daily_range'] / data['prev_range_5d']) - 1
    data['volatility_trend'] = data['volatility_trend'].fillna(0)
    
    # Volatility clustering detection (using rolling standard deviation of ranges)
    data['volatility_clustering'] = data['daily_range'].rolling(window=10, min_periods=5).std()
    data['volatility_clustering'] = data['volatility_clustering'].fillna(data['daily_range'].std())
    
    # 3. Price Level Efficiency Analysis
    # Support/Resistance Context
    data['high_5d'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=3).min()
    
    data['distance_to_high'] = (data['high_5d'] - data['close']) / data['close']
    data['distance_to_low'] = (data['close'] - data['low_5d']) / data['close']
    
    range_5d = data['high_5d'] - data['low_5d']
    range_5d = np.where(range_5d == 0, 1e-10, range_5d)
    data['price_level_efficiency'] = (data['close'] - data['open']) / range_5d
    
    # Level Breakout Confirmation
    data['opening_above_resistance'] = ((data['open'] > data['high_5d']) & 
                                       (data['open'] > data['prev_close'])).astype(int)
    data['opening_below_support'] = ((data['open'] < data['low_5d']) & 
                                    (data['open'] < data['prev_close'])).astype(int)
    
    # 4. Volume-Price Efficiency Framework
    # Volume Momentum Dynamics
    data['volume_acceleration'] = (data['volume'] / data['prev_volume'] - 
                                  data['prev_volume'] / data['prev_volume_2'])
    data['volume_acceleration'] = data['volume_acceleration'].fillna(0)
    
    data['volume_concentration'] = data['volume'] / high_low_range
    data['volume_direction_alignment'] = (np.sign(data['close'] - data['open']) * 
                                         np.sign(data['volume'] - data['prev_volume']))
    
    # 5. Multi-Timeframe Momentum Alignment
    # Short-term vs Medium-term Momentum
    data['momentum_1d'] = (data['close'] - data['prev_close']) / data['prev_close']
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_alignment'] = np.sign(data['momentum_1d']) * np.sign(data['momentum_5d'])
    
    # Opening gap vs trend alignment
    data['gap_trend_alignment'] = (np.sign(data['open'] - data['prev_close']) * 
                                  np.sign(data['momentum_5d']))
    
    # 6. Adaptive Composite Construction
    # Core Momentum Component
    volatility_weight = 1 / (1 + data['volatility_percentile'])  # Inverse weighting for high volatility
    core_momentum = (data['abs_gap_strength'] * data['directional_gap_persistence'] * 
                    volatility_weight)
    
    # Price level efficiency adjustment
    price_efficiency_adj = (data['price_level_efficiency'] * 
                           (1 + data['opening_above_resistance'] - data['opening_below_support']))
    
    # Confirmation Framework
    volume_confirmation = (data['volume_direction_alignment'] * 
                          np.abs(data['volume_acceleration']))
    
    multi_timeframe_confirmation = (data['momentum_alignment'] * data['gap_trend_alignment'])
    
    # Risk Adjustment
    volatility_risk_scaling = 1 / (1 + np.abs(data['volatility_trend']))
    price_level_risk = 1 / (1 + np.abs(data['distance_to_high'] - data['distance_to_low']))
    
    # Final Signal Generation
    final_signal = (core_momentum * price_efficiency_adj * 
                   (1 + 0.5 * volume_confirmation) * 
                   (1 + 0.3 * multi_timeframe_confirmation) * 
                   volatility_risk_scaling * price_level_risk)
    
    # Clean up and return
    result = final_signal.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
