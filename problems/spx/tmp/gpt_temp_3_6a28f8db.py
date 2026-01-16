import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Momentum Component
    df['Raw_Momentum'] = (df['high'] - df['low']) / df['open']
    
    # Volume Adjustment
    df['Rolling_Volume_Mean'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['Volume_Ratio'] = df['volume'] / df['Rolling_Volume_Mean']
    df['Volume_Adjusted_Momentum'] = df['Raw_Momentum'] * df['Volume_Ratio']
    
    # Trend Filter
    df['Trend_Strength'] = abs(df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['Trend_Filtered_Momentum'] = df['Volume_Adjusted_Momentum'] * df['Trend_Strength']
    
    # Volatility Filter
    df['Rolling_Volatility'] = df['close'].rolling(window=30, min_periods=1).std() / df['close']
    df['Volatility_Scaled_Momentum'] = df['Trend_Filtered_Momentum'] * df['Rolling_Volatility'].apply(lambda x: 0.6 if x > df['Rolling_Volatility'].median() else 1.4)
    
    return df['Volatility_Scaled_Momentum']
