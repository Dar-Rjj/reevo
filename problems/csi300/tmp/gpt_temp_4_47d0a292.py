import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate intraday price change
    df['intraday_price_change'] = (df['close'] - df['open']) / df['open']
    
    # Calculate 10-day rolling rank of intraday price change
    df['price_rank'] = df['intraday_price_change'].rolling(10, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Calculate volume deviation (current volume / 10-day median volume)
    df['volume_deviation'] = df['volume'] / df['volume'].rolling(10, min_periods=1).median()
    
    # Calculate 10-day rolling rank of volume deviation
    df['volume_rank'] = df['volume_deviation'].rolling(10, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Combine signals by subtracting volume rank from price rank
    factor = df['price_rank'] - df['volume_rank']
    
    # Rank normalization
    factor = factor.rank(pct=True)
    
    return factor
