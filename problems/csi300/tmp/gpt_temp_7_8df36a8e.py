import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Raw Momentum: EMA of price changes over 5 days with span=10
    delta = df['close'] - df['close'].shift(5)
    raw_momentum = delta.ewm(span=10, adjust=False).mean()
    
    # Cross-sectional rank of raw momentum
    ranked_momentum = raw_momentum.groupby(raw_momentum.index).rank(pct=True)
    
    # Liquidity Adjustment
    # Volume rank over 20 days
    volume_rank = df['volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Volume threshold filter (current volume / 20-day avg volume > 1.2)
    vol_ratio = df['volume'] / df['volume'].rolling(20).mean()
    vol_filter = (vol_ratio > 1.2).astype(int)
    
    # Combine components
    liquidity_adjusted = ranked_momentum * volume_rank * vol_filter
    
    return liquidity_adjusted
