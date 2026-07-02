import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Midday (closing prices at midday)
    midday = df.groupby(df.index.date)['close'].transform(lambda x: x.iloc[len(x)//2])
    
    # Identify Midday Minimum and Maximum
    midday_min = df.groupby(df.index.date)['low'].transform(lambda x: x.iloc[:len(x)//2+1].min())
    midday_max = df.groupby(df.index.date)['high'].transform(lambda x: x.iloc[:len(x)//2+1].max())
    
    # Compute Price Reversal Ratio
    price_reversal_ratio = (df['close'] - midday_min) / (midday_max - midday_min)
    
    return price_reversal_ratio
