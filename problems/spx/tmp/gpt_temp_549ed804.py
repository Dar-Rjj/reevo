import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Intraday Momentum
    intraday_momentum = (df['high'] - df['low']) / df['close']
    
    # Compute 20-day Volume Moving Average
    volume_ma = df['volume'].rolling(window=20, min_periods=1).mean()
    
    # Volume Adjustment
    volume_adjustment = df['volume'] / volume_ma
    
    # Combine components
    factor = intraday_momentum / volume_adjustment
    
    return factor
