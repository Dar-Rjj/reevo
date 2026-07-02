import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Normalized Gap
    df['Normalized_Gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['High_Low_Range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Combine with Volume
    df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['Combined_Gap_Volume'] = df['Normalized_Gap'] * df['Volume_Ratio']
    
    # Calculate Cross-Sectional Rank
    df['Rank'] = df['Combined_Gap_Volume'].rolling(window=len(df), min_periods=1).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Adjust for Volatility
    df['Returns'] = df['close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=5).std()
    df['Factor'] = df['Rank'] / df['Volatility'].replace(0, 1)  # Avoid division by zero
    
    return df['Factor']
