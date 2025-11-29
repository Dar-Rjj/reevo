import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Gap Reversal with Range Persistence
    # Calculate Overnight Gap Magnitude
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)).abs()
    
    # Compute Range Persistence Component
    data['daily_range'] = data['high'] - data['low']
    data['range_change'] = data['daily_range'] - data['daily_range'].shift(1)
    
    # Calculate range direction consistency
    data['range_direction'] = np.sign(data['range_change'])
    data['consecutive_direction'] = 0
    for i in range(1, len(data)):
        if data['range_direction'].iloc[i] == data['range_direction'].iloc[i-1]:
            data.loc[data.index[i], 'consecutive_direction'] = data['consecutive_direction'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'consecutive_direction'] = 1
    
    data['range_persistence'] = data['consecutive_direction'] * data['range_change'].abs()
    
    # Calculate Intraday Reversal Signal
    data['intraday_reversal'] = np.sign(data['close'] - data['open'])
    
    # Combine components for first factor
    factor1 = data['overnight_gap'] * data['range_persistence'] * data['intraday_reversal']
    
    # Volume-Weighted True Range Breakout
    # Calculate Modified True Range
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        )
    )
    
    # Incorporate gap information
    data['modified_true_range'] = data['true_range'] + data['overnight_gap']
    
    # Compute Volume-Weighted Breakout
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_weighted_breakout'] = data['volume_momentum'] * data['modified_true_range']
    
    # Apply Persistence Filter
    data['breakout_3day_avg'] = data['volume_weighted_breakout'].rolling(window=3, min_periods=1).mean()
    data['breakout_direction'] = np.sign(data['volume_weighted_breakout'] - data['breakout_3day_avg'])
    data['breakout_consistency'] = data['breakout_direction'].rolling(window=3, min_periods=1).apply(
        lambda x: len(set(x)) == 1 if len(x) == 3 else 0
    )
    
    data['persistence_weight'] = data['breakout_consistency'] * data['breakout_direction'].abs()
    factor2 = data['volume_weighted_breakout'] * data['persistence_weight'] / data['modified_true_range']
    
    # Intraday Pressure Range Divergence
    # Calculate Buying Pressure Component
    data['buying_pressure'] = (
        (data['close'] - data['open']) / data['open'] + 
        (data['close'] - data['low']) / data['low']
    ) * data['range_persistence']
    
    # Calculate Selling Pressure Component
    data['selling_pressure'] = (
        (data['open'] - data['close']) / data['open'] + 
        (data['high'] - data['close']) / data['high']
    ) * data['overnight_gap']
    
    # Compute Net Pressure Divergence
    data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
    data['pressure_divergence'] = data['net_pressure'] * data['range_persistence'].rolling(window=3, min_periods=1).std()
    
    # Apply Volume Confirmation
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0
    )
    factor3 = data['pressure_divergence'] * data['volume_trend']
    
    # Volatility Clustering with Gap Reversal
    # Calculate Gap-Based Volatility
    data['gap_volatility'] = data['overnight_gap'] * data['daily_range']
    
    # Identify Clustering Patterns
    data['volatility_3period'] = data['gap_volatility'].rolling(window=3, min_periods=1).mean()
    data['clustering_intensity'] = data['gap_volatility'] / data['volatility_3period']
    
    # Compute Reversal Momentum
    data['reversal_momentum'] = (data['close'] - data['open']) / data['open'] * data['intraday_reversal']
    data['weighted_reversal'] = data['reversal_momentum'] * data['clustering_intensity']
    
    # Generate Mean Reversion Factor
    factor4 = -data['gap_volatility'] * data['weighted_reversal']
    
    # Combine all factors with equal weighting
    combined_factor = (factor1.fillna(0) + factor2.fillna(0) + factor3.fillna(0) + factor4.fillna(0)) / 4
    
    return combined_factor
