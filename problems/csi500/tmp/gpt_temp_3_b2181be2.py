import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Factor 1: Intraday Momentum Divergence with Volume Confirmation
    # Calculate Mid-Point Price
    data['mid_point'] = (data['high'] + data['low']) / 2
    
    # Calculate Momentum Signal (current vs 5-day median)
    data['mid_median_5d'] = data['mid_point'].rolling(window=5, min_periods=3).median()
    data['momentum_signal'] = data['mid_point'] - data['mid_median_5d']
    
    # Volume-Weighted Price Extremes
    data['high_vol_weighted'] = data['high'] * data['volume']
    data['low_vol_weighted'] = data['low'] * data['volume']
    data['extreme_ratio'] = data['high_vol_weighted'] / (data['low_vol_weighted'] + 1e-8)
    
    # Combine with Momentum and apply Z-Score
    data['momentum_extreme'] = data['momentum_signal'] * data['extreme_ratio']
    data['divergence_zscore'] = (data['momentum_extreme'] - data['momentum_extreme'].rolling(window=20, min_periods=10).mean()) / (data['momentum_extreme'].rolling(window=20, min_periods=10).std() + 1e-8)
    
    # Volume Confirmation
    data['vol_avg_20d'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['vol_ratio'] = data['volume'] / (data['vol_avg_20d'] + 1e-8)
    data['factor1'] = data['divergence_zscore'] * data['vol_ratio']
    
    # Factor 2: Range Breakout with Momentum Persistence
    # Identify Range Breakouts
    data['high_20d'] = data['high'].rolling(window=20, min_periods=10).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=10).min()
    data['breakout_high'] = (data['close'] > data['high_20d'].shift(1)).astype(int)
    data['breakout_low'] = (data['close'] < data['low_20d'].shift(1)).astype(int)
    data['breakout_flag'] = data['breakout_high'] - data['breakout_low']
    
    # Momentum Persistence Assessment
    data['intraday_return'] = (data['high'] - data['low']) / (data['low'] + 1e-8)
    
    # Track momentum consistency (consecutive same-direction intraday moves)
    data['intraday_direction'] = np.sign(data['intraday_return'])
    data['momentum_persistence'] = 0
    for i in range(1, len(data)):
        if data['intraday_direction'].iloc[i] == data['intraday_direction'].iloc[i-1]:
            data['momentum_persistence'].iloc[i] = data['momentum_persistence'].iloc[i-1] + 1
        else:
            data['momentum_persistence'].iloc[i] = 1
    
    data['breakout_momentum'] = data['breakout_flag'] * data['momentum_persistence'] * data['intraday_return']
    
    # Volume-Weighted Confirmation
    data['breakout_vol_ratio'] = data['volume'] / (data['vol_avg_20d'] + 1e-8)
    data['factor2'] = data['breakout_momentum'] * data['breakout_vol_ratio']
    
    # Factor 3: Volatility-Adjusted Pressure Accumulation
    # Calculate Pressure Accumulation
    data['buying_pressure'] = np.where(data['close'] > data['open'], (data['close'] - data['open']) * data['volume'], 0)
    data['selling_pressure'] = np.where(data['close'] < data['open'], (data['open'] - data['close']) * data['volume'], 0)
    
    data['pressure_balance'] = (data['buying_pressure'] - data['selling_pressure']) / (data['volume'] + 1e-8)
    data['cumulative_pressure'] = data['pressure_balance'].cumsum()
    
    # Adjust for Volatility Regime
    data['price_volatility'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['volatility_10d'] = data['price_volatility'].rolling(window=10, min_periods=5).mean()
    
    data['vol_adjusted_pressure'] = data['cumulative_pressure'] / (data['volatility_10d'] + 1e-8)
    data['vol_adjusted_pressure_smooth'] = data['vol_adjusted_pressure'].ewm(span=5, min_periods=3).mean()
    
    # Combine with Efficiency Signal
    data['abs_return'] = abs(data['close'].pct_change())
    data['efficiency_5d'] = data['abs_return'].rolling(window=5, min_periods=3).sum()
    
    data['efficiency_adjusted_pressure'] = data['vol_adjusted_pressure_smooth'] * data['efficiency_5d']
    data['factor3'] = data['efficiency_adjusted_pressure'].diff(3)
    
    # Factor 4: Gap Persistence with Momentum Divergence
    # Calculate Opening Gap Momentum
    data['prev_close'] = data['close'].shift(1)
    data['opening_gap'] = (data['open'] - data['prev_close']) / (data['prev_close'] + 1e-8)
    
    # Track gap direction consistency
    data['gap_direction'] = np.sign(data['opening_gap'])
    data['gap_persistence'] = 0
    for i in range(1, len(data)):
        if data['gap_direction'].iloc[i] == data['gap_direction'].iloc[i-1]:
            data['gap_persistence'].iloc[i] = data['gap_persistence'].iloc[i-1] + 1
        else:
            data['gap_persistence'].iloc[i] = 1
    
    data['gap_momentum'] = data['gap_persistence'] * data['opening_gap']
    
    # Add Intraday Momentum Divergence
    data['momentum_divergence'] = data['mid_point'] - data['mid_median_5d']
    data['momentum_divergence_extreme'] = data['momentum_divergence'] * data['extreme_ratio']
    
    # Combine Signals with Volume Confirmation
    data['gap_momentum_combined'] = data['gap_momentum'] * data['momentum_divergence_extreme']
    data['gap_vol_confirmed'] = data['gap_momentum_combined'] * data['vol_ratio']
    
    # Apply regime adjustment
    data['volatility_5d'] = data['price_volatility'].rolling(window=5, min_periods=3).mean()
    data['factor4'] = data['gap_vol_confirmed'] / (data['volatility_5d'] + 1e-8)
    
    # Combine all factors with equal weighting
    factors = ['factor1', 'factor2', 'factor3', 'factor4']
    for factor in factors:
        data[factor] = (data[factor] - data[factor].mean()) / (data[factor].std() + 1e-8)
    
    data['combined_factor'] = data[factors].mean(axis=1)
    
    return data['combined_factor']
