import pandas as pd
def heuristics_v2(df):
    # Calculate Price Efficiency
    price_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Calculate Volume Percentile
    volume_rank = df['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )
    
    # Combine Signals
    factor = price_efficiency * volume_rank
    
    return factor
