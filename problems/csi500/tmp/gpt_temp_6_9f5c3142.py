import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Gap components
    data['gap_magnitude'] = np.abs(data['open'] - data['prev_close']) / data['prev_close']
    data['opening_gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Breakout components
    data['range_breakout'] = ((data['close'] > data['prev_high']) | (data['close'] < data['prev_low'])).astype(int)
    data['breakout_magnitude'] = np.abs(data['close'] - data['prev_close']) / data['prev_close']
    
    # Efficiency components
    data['directional_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['absolute_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume and liquidity components
    data['volume_10d_median'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_intensity'] = data['volume'] / data['volume_10d_median']
    data['volume_surge'] = data['volume'] / data['volume_20d_avg']
    
    data['daily_liquidity'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    data['liquidity_5d_avg'] = data['daily_liquidity'].rolling(window=5, min_periods=3).mean()
    data['liquidity_surge'] = data['daily_liquidity'] / data['liquidity_5d_avg']
    data['volume_concentration'] = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volatility components
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['volatility_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['volatility_20d'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    data['volatility_60d'] = data['true_range'].rolling(window=60, min_periods=30).mean()
    data['volatility_ratio'] = data['volatility_20d'] / data['volatility_60d']
    
    # Volume trends and acceleration
    data['volume_5d_ma'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_20d_ma'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_trend'] = data['volume_5d_ma'] / data['volume_20d_ma']
    
    data['volume_3d_return'] = data['volume'].pct_change(3)
    data['volume_10d_return'] = data['volume'].pct_change(10)
    data['volume_acceleration'] = data['volume_3d_return'] - data['volume_10d_return']
    
    # Liquidity trends
    data['liquidity_5d_ma'] = data['daily_liquidity'].rolling(window=5, min_periods=3).mean()
    data['liquidity_20d_ma'] = data['daily_liquidity'].rolling(window=20, min_periods=10).mean()
    data['liquidity_trend'] = data['liquidity_5d_ma'] / data['liquidity_20d_ma']
    
    # Price returns and acceleration
    data['price_3d_return'] = data['close'].pct_change(3)
    data['price_10d_return'] = data['close'].pct_change(10)
    data['price_acceleration'] = data['price_3d_return'] - data['price_10d_return']
    
    # Efficiency momentum
    data['efficiency_3d'] = data['directional_efficiency'].rolling(window=3, min_periods=2).mean()
    data['efficiency_10d'] = data['directional_efficiency'].rolling(window=10, min_periods=5).mean()
    data['efficiency_change'] = data['efficiency_3d'] - data['efficiency_10d']
    
    # Calculate persistence components
    def consecutive_days(series, condition):
        counter = series.copy()
        counter.iloc[0] = 1 if condition.iloc[0] else 0
        for i in range(1, len(series)):
            if condition.iloc[i]:
                counter.iloc[i] = counter.iloc[i-1] + 1
            else:
                counter.iloc[i] = 0
        return counter
    
    # Breakout persistence
    data['consecutive_breakout_days'] = consecutive_days(data['range_breakout'], data['range_breakout'] == 1)
    
    # High efficiency persistence
    high_efficiency = data['absolute_efficiency'] > 0.7
    data['consecutive_high_efficiency_days'] = consecutive_days(data['absolute_efficiency'], high_efficiency)
    
    # Same direction efficiency persistence
    efficiency_direction = data['directional_efficiency'] > 0
    data['consecutive_efficiency_direction'] = consecutive_days(data['directional_efficiency'], efficiency_direction)
    
    # Gap-efficiency persistence
    same_direction_gap = (data['opening_gap'] * data['directional_efficiency']) > 0
    data['consecutive_gap_efficiency_days'] = consecutive_days(data['opening_gap'], same_direction_gap)
    
    # High-efficiency gap days
    high_efficiency_gap = (data['absolute_efficiency'] > 0.7) & (np.abs(data['opening_gap']) > 0.01)
    data['consecutive_high_efficiency_gap_days'] = consecutive_days(data['absolute_efficiency'], high_efficiency_gap)
    
    # Breakout-liquidity alignment
    breakout_liquidity_support = (data['range_breakout'] == 1) & (data['liquidity_surge'] > 1)
    data['consecutive_breakout_liquidity_days'] = consecutive_days(data['range_breakout'], breakout_liquidity_support)
    
    # Calculate breakout-liquidity correlation
    data['breakout_liquidity_product'] = data['range_breakout'] * data['liquidity_surge']
    data['breakout_liquidity_correlation'] = data['breakout_liquidity_product'].rolling(window=10, min_periods=5).mean()
    
    # Calculate efficiency-breakout covariance
    data['efficiency_breakout_product'] = data['directional_efficiency'] * data['breakout_magnitude']
    data['efficiency_breakout_covariance'] = data['efficiency_breakout_product'].rolling(window=10, min_periods=5).mean()
    
    # Factor 1: Intraday Gap-Breakout Efficiency Composite
    gap_breakout_efficiency = (
        data['gap_magnitude'] * 
        data['breakout_magnitude'] * 
        data['directional_efficiency'] * 
        data['volume_intensity'] * 
        data['volume_concentration'] * 
        data['consecutive_breakout_days'] * 
        data['consecutive_high_efficiency_days']
    )
    
    # Factor 2: Volatility-Adjusted Efficiency Momentum
    volatility_adjusted_efficiency = (
        data['efficiency_change'] * 
        data['consecutive_efficiency_direction'] * 
        (1 / data['volatility_10d']) * 
        data['volatility_ratio'] * 
        data['volume_trend'] * 
        data['volume_acceleration']
    )
    
    # Factor 3: Liquidity-Breakout Divergence Factor
    liquidity_breakout_divergence = (
        data['breakout_liquidity_correlation'] * 
        data['consecutive_breakout_liquidity_days'] * 
        data['price_acceleration'] * 
        data['directional_efficiency'] * 
        data['liquidity_surge']
    )
    
    # Factor 4: Volume-Weighted Gap Efficiency Factor
    volume_weighted_gap_efficiency = (
        data['opening_gap'] * 
        data['directional_efficiency'] * 
        data['volume_surge'] * 
        data['volume_concentration'] * 
        data['consecutive_gap_efficiency_days'] * 
        data['consecutive_high_efficiency_gap_days']
    )
    
    # Factor 5: Breakout-Efficiency Correlation Momentum
    breakout_efficiency_momentum = (
        data['efficiency_breakout_covariance'] * 
        data['volume_trend'] * 
        data['volume_acceleration'] * 
        data['price_acceleration'] * 
        data['efficiency_change'] * 
        data['consecutive_breakout_days']
    )
    
    # Factor 6: Liquidity-Efficiency Breakout Composite
    liquidity_efficiency_breakout = (
        data['daily_liquidity'] * 
        data['range_breakout'] * 
        data['directional_efficiency'] * 
        data['gap_magnitude'] * 
        data['consecutive_breakout_days'] * 
        data['consecutive_high_efficiency_days'] * 
        data['volume_surge'] * 
        data['liquidity_trend']
    )
    
    # Combine all factors with equal weighting
    combined_factor = (
        gap_breakout_efficiency + 
        volatility_adjusted_efficiency + 
        liquidity_breakout_divergence + 
        volume_weighted_gap_efficiency + 
        breakout_efficiency_momentum + 
        liquidity_efficiency_breakout
    )
    
    # Return the combined factor series
    return combined_factor
