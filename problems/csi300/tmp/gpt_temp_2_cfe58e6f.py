import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Price Change
    price_change = df['close'] / df['close'].shift(1) - 1
    
    # Normalize by Trading Range
    trading_range = df['high'] - df['low']
    price_efficiency = (df['close'] - df['open']) / trading_range
    
    # Calculate Volume Percentile
    volume_percentile = df['volume'].rolling(window=20).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    # Apply Volume Filter
    factor = price_efficiency * volume_percentile
    
    return factor
