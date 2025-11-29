import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Identify Intraday Breakout Strength
    # Compute Opening Gap Magnitude
    data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Calculate Breakout Intensity
    # Compute Current Day Range Utilization
    data['range_utilization'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['range_utilization'] = data['range_utilization'].replace([np.inf, -np.inf], np.nan)
    
    data['breakout_strength'] = data['gap'] * data['range_utilization']
    
    # 2. Detect Adaptive Trend Regimes
    # Calculate Dual-Timeframe Trend Direction
    data['st_trend'] = np.sign(data['close'] - data['close'].shift(3))
    data['mt_trend'] = np.sign(data['close'] - data['close'].shift(10))
    
    # Determine Trend Regime
    data['trend_regime'] = data['st_trend'] == data['mt_trend']
    
    # 3. Compute Volume-Weighted Reversal Signal
    # Calculate Relative Volume Activity
    data['volume_percentile'] = data['volume'].rolling(window=15, min_periods=1).apply(
        lambda x: (x.rank().iloc[-1] / len(x)) * 100 if len(x) > 0 else np.nan
    )
    data['volume_weight'] = data['volume_percentile'] / 100
    
    # Generate Reversal Indicator
    data['price_reversal'] = (data['high'] + data['low']) / 2 - data['close'].shift(1)
    data['previous_range'] = data['high'].shift(1) - data['low'].shift(1)
    data['scaled_reversal'] = data['price_reversal'] / data['previous_range']
    data['scaled_reversal'] = data['scaled_reversal'].replace([np.inf, -np.inf], np.nan)
    
    data['volume_adjusted_reversal'] = data['scaled_reversal'] * data['volume_weight']
    
    # 4. Apply Regime-Based Signal Combination
    data['signal'] = np.nan
    # Aligned Trend Regime
    aligned_mask = data['trend_regime'] == True
    data.loc[aligned_mask, 'signal'] = (
        data.loc[aligned_mask, 'breakout_strength'] * 
        data.loc[aligned_mask, 'volume_adjusted_reversal']
    )
    
    # Conflicting Trend Regime
    conflicting_mask = data['trend_regime'] == False
    data.loc[conflicting_mask, 'signal'] = (
        -data.loc[conflicting_mask, 'breakout_strength'] * 
        data.loc[conflicting_mask, 'volume_adjusted_reversal']
    )
    
    # 5. Generate Final Alpha Factor
    # Calculate Signal Persistence
    def calculate_consistency(series):
        if len(series) < 5:
            return np.nan
        current_sign = np.sign(series.iloc[-1])
        past_signs = np.sign(series.iloc[-5:-1])
        consistency = np.sum(past_signs == current_sign)
        return consistency
    
    data['consistency'] = data['signal'].rolling(window=5, min_periods=1).apply(
        calculate_consistency, raw=False
    )
    data['persistence_weight'] = data['consistency'] / 5
    
    # Final Factor
    data['final_factor'] = data['signal'] * data['persistence_weight']
    
    return data['final_factor']
