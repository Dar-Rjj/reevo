import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate previous close for True Range calculation
    data['Close_prev'] = data['close'].shift(1)
    
    # 1. Calculate True Range
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['Close_prev']),
            abs(data['low'] - data['Close_prev'])
        )
    )
    
    # 2. Calculate Price Efficiency Ratio
    data['Net_Price_Change'] = abs(data['close'] - data['open'])
    data['Efficiency_Ratio'] = data['Net_Price_Change'] / data['TR']
    data['Efficiency_Ratio'] = data['Efficiency_Ratio'].replace([np.inf, -np.inf], np.nan)
    
    # 3. Calculate Volume-Weighted Efficiency
    data['VW_Movement'] = (data['close'] - data['open']) * data['volume']
    data['Absolute_VW_Movement'] = abs(data['VW_Movement'])
    data['Volume_Efficiency'] = data['Absolute_VW_Movement'] / (data['TR'] * data['volume'])
    data['Volume_Efficiency'] = data['Volume_Efficiency'].replace([np.inf, -np.inf], np.nan)
    data['Efficiency_Divergence'] = data['Efficiency_Ratio'] - data['Volume_Efficiency']
    
    # 4. Calculate Volume Acceleration Pattern
    data['Volume_shift5'] = data['volume'].shift(5)
    data['Volume_shift10'] = data['volume'].shift(10)
    data['Volume_Change_5d'] = data['volume'] / data['Volume_shift5']
    data['Volume_Change_10d'] = data['volume'] / data['Volume_shift10']
    data['Volume_Acceleration'] = data['Volume_Change_5d'] - data['Volume_Change_10d']
    
    # Determine Acceleration Regime
    def get_acceleration_multiplier(accel):
        if accel > 0.1:
            return 1.5
        elif accel < -0.1:
            return 0.5
        else:
            return 1.0
    
    data['Acceleration_Multiplier'] = data['Volume_Acceleration'].apply(get_acceleration_multiplier)
    
    # 5. Calculate Price Compression State
    # Calculate 14-day ATR
    data['ATR_14'] = data['TR'].rolling(window=14, min_periods=1).mean()
    data['Compression_Ratio'] = data['TR'] / data['ATR_14']
    
    # Calculate Intraday Bias
    data['Bias'] = (data['close'] - data['open']) / data['TR']
    data['Bias'] = data['Bias'].replace([np.inf, -np.inf], np.nan)
    data['Absolute_Bias'] = abs(data['Bias'])
    data['Compression_Bias_Score'] = data['Bias'] * (1 - data['Compression_Ratio'])
    
    # 6. Detect Volume-Price Divergence Strength
    # Calculate 5-day price and volume momentum
    data['Close_shift5'] = data['close'].shift(5)
    data['Price_Momentum_5d'] = (data['close'] - data['Close_shift5']) / data['Close_shift5']
    data['Volume_Momentum_5d'] = (data['volume'] - data['Volume_shift5']) / data['Volume_shift5']
    data['Divergence'] = data['Price_Momentum_5d'] - data['Volume_Momentum_5d']
    
    # Calculate 5-day volume-price correlation
    data['Price_Change_5d'] = data['close'].pct_change(periods=5)
    data['Volume_Change_5d_pct'] = data['volume'].pct_change(periods=5)
    
    # Rolling correlation over 5 days
    corr_window = 5
    corr_values = []
    for i in range(len(data)):
        if i < corr_window - 1:
            corr_values.append(np.nan)
        else:
            window_data = data.iloc[i-corr_window+1:i+1]
            corr = window_data['Price_Change_5d'].corr(window_data['Volume_Change_5d_pct'])
            corr_values.append(corr if not np.isnan(corr) else 0)
    
    data['Corr_5d'] = corr_values
    
    # Calculate Divergence Persistence
    divergence_sign = np.sign(data['Divergence'])
    persistence_scores = []
    for i in range(len(data)):
        if i < 2:
            persistence_scores.append(0)
        else:
            current_sign = divergence_sign.iloc[i]
            prev_signs = divergence_sign.iloc[i-2:i]
            same_sign_count = (prev_signs == current_sign).sum()
            persistence_scores.append(same_sign_count / 3)
    
    data['Persistence_Score'] = persistence_scores
    data['Persistent_Divergence'] = data['Divergence'] * data['Persistence_Score']
    
    # 7. Generate Final Alpha Factor
    # Combine Efficiency and Acceleration
    data['Efficiency_Acceleration'] = data['Efficiency_Divergence'] * data['Volume_Acceleration'] * data['Acceleration_Multiplier']
    
    # Apply Compression State Adjustment
    data['Compression_Adjusted'] = data['Efficiency_Acceleration'] * data['Compression_Bias_Score']
    
    # Incorporate Divergence Strength
    corr_weight = 1 - abs(data['Corr_5d'])
    data['Divergence_Enhanced'] = data['Compression_Adjusted'] * data['Persistent_Divergence'] * corr_weight
    
    # Apply Intraday Range Normalization
    data['Range_Efficiency'] = data['Efficiency_Ratio'] * data['TR']
    data['Final_Factor'] = data['Divergence_Enhanced'] * data['Range_Efficiency']
    
    # Return the final factor series
    return data['Final_Factor']
