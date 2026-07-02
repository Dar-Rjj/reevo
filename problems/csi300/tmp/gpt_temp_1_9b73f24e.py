import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute True Range
    true_range = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    
    # Compute Closing Range Utilization
    closing_range = (df['close'] - df['open']) / true_range
    
    # Compute Volume Percentile
    volume_rank = df['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    min_rank = volume_rank.rolling(window=20).min()
    max_rank = volume_rank.rolling(window=20).max()
    volume_percentile = (volume_rank - min_rank) / (max_rank - min_rank)
    
    # Combine Signals
    factor = closing_range * volume_percentile
    
    return factor
