import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Normalized Daily Range
    df['Normalized_Range'] = (df['high'] - df['low']) / df['close']
    
    # 5-day Z-Score of Normalized Range
    df['Z_Score'] = df['Normalized_Range'].rolling(window=5).apply(lambda x: zscore(x)[-1], raw=True)
    
    # Flip Sign for Reversal
    df['Reversal'] = -1 * df['Z_Score']
    
    # Compute Volume Spike
    df['MA_Volume_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Spike'] = df['volume'] / df['MA_Volume_20']
    
    # Adjust Reversal by Volume
    df['Volume_Weighted_Reversal'] = df['Reversal'] * df['Volume_Spike']
    
    # Calculate Price Volatility
    df['Volatility'] = df['close'].rolling(window=20).std() / df['close']
    
    # Final Adjustment
    df['Final_Factor'] = df['Volume_Weighted_Reversal'] / df['Volatility']
    
    return df['Final_Factor']
