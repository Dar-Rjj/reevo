import pandas as pd
def heuristics_v2(df):
    # Raw Momentum: EMA of price change over 20 days
    delta = df['close'] - df['close'].shift(20)
    raw_momentum = delta.ewm(span=20, adjust=False).mean()
    ranked_momentum = raw_momentum.groupby(raw_momentum.index).rank(pct=True)
    
    # Volume Adjustment: rolling rank of volume over 20 days
    volume_rank = df['volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combine momentum with volume adjustment
    factor = ranked_momentum * volume_rank
    
    return factor
