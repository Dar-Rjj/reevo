import pandas as pd
import pandas as pd
from scipy.stats import zscore

def heuristics_v2(df):
    # Momentum Component
    df['Price_Momentum'] = df['close'] / df['close'].shift(1)
    df['Norm_Momentum'] = df['Price_Momentum'].rolling(window=10).apply(lambda x: zscore(x)[-1], raw=True)
    
    # Efficiency Component
    df['Efficiency_Ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['Norm_Efficiency'] = df['Efficiency_Ratio'].rolling(window=10).apply(lambda x: zscore(x)[-1], raw=True)
    
    # Volume Component
    df['Volume_Spike'] = df['volume'] / df['volume'].rolling(window=10).mean()
    df['Norm_Volume'] = df['Volume_Spike'].rolling(window=10).apply(lambda x: zscore(x)[-1], raw=True)
    
    # Combined Signal
    df['Combined_Signal'] = df['Norm_Momentum'] * df['Norm_Efficiency'] * df['Norm_Volume']
    df['Norm_Signal'] = df['Combined_Signal'].rolling(window=10).apply(lambda x: zscore(x)[-1], raw=True)
    
    return df['Norm_Signal']
