import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic components
    df['AM_Momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['PM_Momentum'] = (df['close'] - (df['high'] + df['low']) / 2) / (df['high'] - df['low'])
    
    # Momentum Divergence Component
    df['Momentum_Divergence_Signal'] = abs(df['AM_Momentum'] - df['PM_Momentum'])
    
    # Calculate 3-day rolling correlation between AM and PM momentum
    rolling_corr = []
    for i in range(len(df)):
        if i < 2:
            rolling_corr.append(np.nan)
        else:
            window_am = df['AM_Momentum'].iloc[i-2:i+1]
            window_pm = df['PM_Momentum'].iloc[i-2:i+1]
            if len(window_am.dropna()) >= 2 and len(window_pm.dropna()) >= 2:
                corr = window_am.corr(window_pm)
                rolling_corr.append(corr if not np.isnan(corr) else 0)
            else:
                rolling_corr.append(0)
    df['Rolling_Corr_AM_PM'] = rolling_corr
    df['Momentum_Divergence_Signal'] = df['Momentum_Divergence_Signal'] * df['Rolling_Corr_AM_PM']
    
    # Volume Acceleration Component
    df['Volume_Intensity'] = df['volume'] / df['volume'].rolling(window=15, min_periods=1).median()
    
    # Calculate volume velocity components
    df['Close_Open_Sign'] = np.sign(df['close'] - df['open'])
    df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=3, min_periods=1).mean()
    
    # Calculate 5-day rolling sum of (Close - Open) * Volume
    df['Close_Open_Volume'] = (df['close'] - df['open']) * df['volume']
    df['Rolling_Sum_COV'] = df['Close_Open_Volume'].rolling(window=5, min_periods=1).sum()
    
    df['Volume_Velocity'] = df['Close_Open_Sign'] * df['Volume_Ratio'] * df['Rolling_Sum_COV']
    
    # Reversal-Breakout Component
    df['Rolling_Max_High'] = df['high'].rolling(window=3, min_periods=1).max()
    df['Rolling_Min_Low'] = df['low'].rolling(window=3, min_periods=1).min()
    
    df['High_Rejection'] = (df['Rolling_Max_High'] - df['close']) / df['Rolling_Max_High']
    df['Low_Rejection'] = (df['close'] - df['Rolling_Min_Low']) / df['Rolling_Min_Low']
    
    df['High_Low_Range'] = df['high'] - df['low']
    df['Rolling_Min_Range'] = df['High_Low_Range'].rolling(window=5, min_periods=1).min()
    df['Rolling_Max_Range'] = df['High_Low_Range'].rolling(window=5, min_periods=1).max()
    
    df['Range_Normalized'] = (df['High_Low_Range'] - df['Rolling_Min_Range']) / (df['Rolling_Max_Range'] - df['Rolling_Min_Range'])
    df['Net_Reversal_Signal'] = (df['Low_Rejection'] - df['High_Rejection']) * df['Range_Normalized']
    
    # Volatility Scaling Component
    df['Prev_Close'] = df['close'].shift(1)
    df['TR1'] = df['high'] - df['low']
    df['TR2'] = abs(df['high'] - df['Prev_Close'])
    df['TR3'] = abs(df['low'] - df['Prev_Close'])
    df['True_Range'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    df['Volatility_Scaling'] = df['True_Range'] / df['True_Range'].rolling(window=20, min_periods=1).median()
    
    # Factor Integration
    df['Core_Momentum_Divergence'] = df['Momentum_Divergence_Signal'] * df['Volume_Velocity']
    df['Volume_Confirmed_Reversal'] = df['Net_Reversal_Signal'] * df['Volume_Intensity']
    
    df['Final_Factor'] = df['Core_Momentum_Divergence'] * df['Volume_Confirmed_Reversal'] * df['Volatility_Scaling']
    
    # Return the final factor series
    return df['Final_Factor']
