import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Divergence Momentum
    # Calculate Intraday Momentum Components
    high_momentum = (data['high'] - data['high'].shift(1)) / data['high'].shift(1)
    low_momentum = (data['low'] - data['low'].shift(1)) / data['low'].shift(1)
    
    # Compute Momentum Divergence
    momentum_divergence = high_momentum - low_momentum
    daily_range = data['high'] - data['low']
    scaled_divergence = momentum_divergence / (daily_range + 1e-8)
    
    # Measure Persistence Pattern
    divergence_lag1 = scaled_divergence.shift(1)
    divergence_lag2 = scaled_divergence.shift(2)
    divergence_lag3 = scaled_divergence.shift(3)
    
    autocorr_1 = scaled_divergence.rolling(window=5).corr(divergence_lag1)
    autocorr_2 = scaled_divergence.rolling(window=5).corr(divergence_lag2)
    autocorr_3 = scaled_divergence.rolling(window=5).corr(divergence_lag3)
    
    divergence_persistence = (autocorr_1 + autocorr_2 + autocorr_3) / 3
    
    # Volume-Weighted Confirmation
    volume_trend = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_ratio = data['volume'] / (data['volume'].shift(1) + 1e-8)
    
    factor_1 = divergence_persistence * volume_trend * volume_ratio
    
    # Opening Gap Range Efficiency
    # Calculate Opening Gap
    opening_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Measure Gap-Range Relationship
    daily_range_2 = data['high'] - data['low']
    gap_range_ratio = opening_gap / (daily_range_2 / data['close'] + 1e-8)
    
    # Track Efficiency Decay
    gap_closure = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    decay_slope = gap_closure.rolling(window=3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Volume Acceleration Integration
    volume_momentum = data['volume'].pct_change()
    volume_acceleration = volume_momentum.diff()
    
    factor_2 = decay_slope * volume_acceleration * gap_range_ratio
    
    # Price-Volume Divergence Persistence
    # Calculate Trend Components
    price_trend = data['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_trend_2 = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    # Measure Divergence Strength
    trend_divergence = price_trend - volume_trend_2
    intraday_volatility = (data['high'] - data['low']) / data['close']
    scaled_divergence_2 = trend_divergence / (intraday_volatility + 1e-8)
    
    # Track Persistence Pattern
    div_lag1 = scaled_divergence_2.shift(1)
    div_lag2 = scaled_divergence_2.shift(2)
    
    autocorr_div_1 = scaled_divergence_2.rolling(window=5).corr(div_lag1)
    autocorr_div_2 = scaled_divergence_2.rolling(window=5).corr(div_lag2)
    
    divergence_persistence_2 = (autocorr_div_1 + autocorr_div_2) / 2
    
    # Amount-Based Confirmation
    dollar_volume_efficiency = data['close'].pct_change() / (data['amount'] + 1e-8)
    
    factor_3 = divergence_persistence_2 * dollar_volume_efficiency
    
    # Multi-Timeframe Range Momentum
    # Calculate Range Components
    short_term_range = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / data['close']
    long_term_range = (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()) / data['close']
    
    short_range_momentum = short_term_range.pct_change()
    long_range_momentum = long_term_range.pct_change()
    
    # Compute Range Ratio Divergence
    range_ratio_divergence = short_range_momentum - long_range_momentum
    
    # Measure Breakout Persistence
    range_lag1 = range_ratio_divergence.shift(1)
    range_lag2 = range_ratio_divergence.shift(2)
    
    autocorr_range_1 = range_ratio_divergence.rolling(window=5).corr(range_lag1)
    autocorr_range_2 = range_ratio_divergence.rolling(window=5).corr(range_lag2)
    
    breakout_persistence = (autocorr_range_1 + autocorr_range_2) / 2
    
    # Volume-Weighted Breakout
    volume_trend_3 = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    
    factor_4 = breakout_persistence * volume_trend_3
    
    # Close Location Efficiency Momentum
    # Calculate CLV Components
    clv = (2 * data['close'] - data['high'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    clv_momentum = clv.diff()
    clv_acceleration = clv_momentum.diff()
    
    # Track Efficiency Pattern
    clv_lag1 = clv_momentum.shift(1)
    clv_lag2 = clv_momentum.shift(2)
    
    autocorr_clv_1 = clv_momentum.rolling(window=5).corr(clv_lag1)
    autocorr_clv_2 = clv_momentum.rolling(window=5).corr(clv_lag2)
    
    clv_persistence = (autocorr_clv_1 + autocorr_clv_2) / 2
    
    # Combine with Volume Dynamics
    volume_acceleration_2 = data['volume'].diff().diff()
    
    # Range-Scaled Signal
    final_clv_signal = clv_persistence * clv_acceleration * volume_acceleration_2
    range_scaled = final_clv_signal / (daily_range / data['close'] + 1e-8)
    
    factor_5 = range_scaled
    
    # Combine all factors with equal weighting
    combined_factor = (factor_1.fillna(0) + factor_2.fillna(0) + factor_3.fillna(0) + 
                      factor_4.fillna(0) + factor_5.fillna(0)) / 5
    
    return combined_factor
