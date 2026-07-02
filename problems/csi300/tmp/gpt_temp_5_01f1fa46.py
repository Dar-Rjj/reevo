import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Recent Return
    df['Recent_Return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Rolling StdDev of Returns (5 days)
    df['Rolling_StdDev_5'] = df['Recent_Return'].rolling(window=5).std()
    
    # Normalize by Volatility
    df['Normalized_Reversal_Signal'] = df['Recent_Return'] / df['Rolling_StdDev_5']
    
    # Calculate True Range
    df['True_Range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1).loc[x.name]), abs(x['low'] - df['close'].shift(1).loc[x.name])), axis=1)
    
    # Calculate Rolling Average of True Range (20 days)
    df['Rolling_Avg_True_Range_20'] = df['True_Range'].rolling(window=20).mean()
    
    # Normalize by Historical Average
    df['Normalized_Breakout_Intensity'] = df['True_Range'] / df['Rolling_Avg_True_Range_20']
    
    # Combine Reversal Signal with Breakout Intensity
    df['Normalized_Reversal_Breakout_Intensity'] = df['Normalized_Reversal_Signal'] * df['Normalized_Breakout_Intensity']
    
    # Return the final factor values
    return df['Normalized_Reversal_Breakout_Intensity']
