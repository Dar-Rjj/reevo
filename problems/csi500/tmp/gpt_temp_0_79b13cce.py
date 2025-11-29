import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    if len(data) < 20:
        return factor_values
    
    # Intraday Momentum Decay
    data['intraday_range'] = data['high'] - data['low']
    data['closing_position'] = (data['close'] - data['low']) / (data['intraday_range'] + 1e-8)  # Avoid division by zero
    data['prev_close'] = data['close'].shift(1)
    data['momentum_factor'] = data['closing_position'] * ((data['open'] / (data['prev_close'] + 1e-8)) - 1) * data['intraday_range']
    
    # Volume-Adjusted Price Reversal
    data['raw_return'] = data['close'] / data['prev_close'] - 1
    data['volume_avg_20d'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_ratio'] = data['volume'] / (data['volume_avg_20d'] + 1e-8)
    data['reversal_signal'] = -1 * (data['raw_return'] * data['volume_ratio'])
    
    # Liquidity Gap Acceleration
    data['spread'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2 + 1e-8)
    data['close_3d_ago'] = data['close'].shift(3)
    data['velocity'] = data['close'] / (data['close_3d_ago'] + 1e-8) - 1
    data['liquidity_factor'] = data['spread'] * data['velocity'] * np.exp(-data['spread'])
    
    # Opening Gap Persistence
    data['opening_gap'] = data['open'] / data['prev_close'] - 1
    data['intraday_strength'] = (data['high'] - data['open']) / (data['open'] + 1e-8)
    data['persistence_factor'] = np.sign(data['opening_gap']) * np.abs(data['opening_gap'] * data['intraday_strength'])
    
    # Volatility-Regulated Trend Following
    data['close_5d_ago'] = data['close'].shift(5)
    data['trend'] = data['close'] / (data['close_5d_ago'] + 1e-8) - 1
    data['range_10d_avg'] = (data['high'] - data['low']).rolling(window=10, min_periods=5).mean()
    data['volatility_ratio'] = data['intraday_range'] / (data['range_10d_avg'] + 1e-8)
    data['trend_factor'] = (data['trend'] / (data['volatility_ratio'] + 1e-8)) * (1 - np.exp(-data['volatility_ratio']))
    
    # Combine all factors with equal weighting
    factors = ['momentum_factor', 'reversal_signal', 'liquidity_factor', 'persistence_factor', 'trend_factor']
    
    # Calculate z-scores for each factor and combine
    for i, date in enumerate(data.index):
        if i >= 20:  # Ensure we have enough history for normalization
            current_data = data.iloc[:i+1]  # Only use data up to current date
            
            combined_factor = 0
            for factor in factors:
                if factor in current_data.columns:
                    factor_data = current_data[factor].dropna()
                    if len(factor_data) > 0:
                        # Calculate z-score using historical data only
                        mean_val = factor_data.mean()
                        std_val = factor_data.std()
                        if std_val > 0:
                            z_score = (current_data[factor].iloc[i] - mean_val) / std_val
                            combined_factor += z_score
            
            factor_values.iloc[i] = combined_factor
    
    return factor_values
