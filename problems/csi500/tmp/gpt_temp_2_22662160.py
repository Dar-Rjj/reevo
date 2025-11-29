import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Momentum Divergence Component
    data['AM_Momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['PM_Momentum'] = (data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Calculate 3-day rolling correlation
    am_momentum_rolling = data['AM_Momentum'].rolling(window=3, min_periods=3)
    pm_momentum_rolling = data['PM_Momentum'].rolling(window=3, min_periods=3)
    
    momentum_corr = []
    for i in range(len(data)):
        if i < 2:
            momentum_corr.append(np.nan)
        else:
            window_am = data['AM_Momentum'].iloc[i-2:i+1]
            window_pm = data['PM_Momentum'].iloc[i-2:i+1]
            valid_mask = (~window_am.isna()) & (~window_pm.isna())
            if valid_mask.sum() >= 2:
                corr_val = window_am[valid_mask].corr(window_pm[valid_mask])
                momentum_corr.append(corr_val if not np.isnan(corr_val) else 0)
            else:
                momentum_corr.append(0)
    
    data['Momentum_Correlation'] = momentum_corr
    data['Momentum_Divergence_Signal'] = abs(data['AM_Momentum'] - data['PM_Momentum']) * data['Momentum_Correlation']
    
    # Volume Acceleration Component
    data['Volume_1d_ago'] = data['volume'].shift(1)
    data['Volume_2d_ago'] = data['volume'].shift(2)
    data['Volume_Ratio'] = data['volume'] / data['Volume_1d_ago']
    data['Volume_Acceleration'] = data['Volume_Ratio'] - (data['Volume_1d_ago'] / data['Volume_2d_ago'])
    data['Volume_Surge_Intensity'] = data['Volume_Ratio'] * data['Volume_Acceleration']
    
    # Volatility Breakout Component
    data['Daily_Range'] = data['high'] - data['low']
    data['Range_10d_Mean'] = data['Daily_Range'].rolling(window=10, min_periods=10).mean()
    data['Range_Compression'] = data['Daily_Range'] / data['Range_10d_Mean']
    
    data['Close_5d_Std'] = data['close'].rolling(window=5, min_periods=5).std()
    data['Close_20d_Std'] = data['close'].rolling(window=20, min_periods=20).std()
    data['Volatility_Ratio'] = data['Close_5d_Std'] / data['Close_20d_Std']
    
    data['Close_5d_Mean'] = data['close'].rolling(window=5, min_periods=5).mean()
    data['Breakout_Signal'] = (data['close'] - data['Close_5d_Mean']) / data['Close_5d_Std'] * (1 / (data['Volatility_Ratio'] * data['Range_Compression'] + 0.001))
    
    # Intraday Efficiency Component
    data['Intraday_Efficiency_Ratio'] = abs(data['close'] - data['open']) / data['Daily_Range'].replace(0, np.nan)
    data['Efficiency_Strength'] = data['Intraday_Efficiency_Ratio'] * data['Daily_Range'] * data['Volume_Ratio']
    
    # Factor Integration
    data['Core_Divergence'] = data['Momentum_Divergence_Signal'] * data['Volume_Surge_Intensity']
    data['Breakout_Strength'] = data['Breakout_Signal'] * data['Efficiency_Strength']
    data['Final_Factor'] = data['Core_Divergence'] * data['Breakout_Strength']
    
    # Return the final factor series
    return data['Final_Factor']
