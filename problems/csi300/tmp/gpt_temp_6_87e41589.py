import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Return
    df['Return'] = df['high'] / df['low'].shift(1) - 1
    
    # Smoothed Momentum
    df['Smoothed_Momentum'] = df['Return'].rolling(window=10, min_periods=1).mean()
    
    # Normalize Smoothed Momentum cross-sectionally
    df['Normalized_Momentum'] = df.groupby(df.index)['Smoothed_Momentum'].transform(lambda x: (x - x.mean()) / x.std())
    
    # Decay Factor
    decay_rate = 0.8
    time_window = 15
    df['Decay_Factor'] = np.exp(-decay_rate * np.arange(time_window)[::-1])
    
    # Momentum Decay Strength
    df['Momentum_Decay_Strength'] = df['Normalized_Momentum'] * df['Decay_Factor'].iloc[-1]
    
    return df['Momentum_Decay_Strength']
