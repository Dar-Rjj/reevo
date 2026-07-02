import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Gap Measurement
    df['Close_prev'] = df['close'].shift(1)
    df['Gap_Size'] = (df['open'] - df['Close_prev']) / df['Close_prev']
    
    # Relative Gap Strength
    df['Open_Close_Ratio'] = df['open'] / df['Close_prev']
    df['Gap_Strength_STD'] = df['Open_Close_Ratio'].rolling(window=10).std()
    df['Relative_Gap_Strength'] = df['Gap_Size'] / df['Gap_Strength_STD']
    
    # Intraday Confirmation - Directional Consistency
    df['Intraday_Move'] = (df['close'] - df['open']) / df['open']
    df['Directional_Consistency'] = np.sign(df['Gap_Size']) * df['Intraday_Move']
    
    # Volume Support
    df['SMA_Volume_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Support'] = df['volume'] / df['SMA_Volume_20']
    
    # Signal Construction
    df['Combined_Signal'] = df['Relative_Gap_Strength'] * df['Directional_Consistency'] * df['Volume_Support']
    
    # Normalization
    df['Heuristic_Factor'] = df['Combined_Signal'].groupby(df.index).transform(lambda x: (x - x.mean()) / x.std())
    
    return df['Heuristic_Factor']
