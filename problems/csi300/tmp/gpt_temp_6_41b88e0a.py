import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Raw Momentum - EMA of price changes over 5 days with span 10
    momentum = df['close'].diff(5).ewm(span=10, adjust=False).mean()
    
    # Cross-sectional rank of the momentum
    ranked_momentum = momentum.groupby(momentum.index).rank(pct=True)
    
    # Liquidity Adjustment - rolling rank of volume over 20 days
    liquidity = df['volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combine momentum and liquidity
    factor = ranked_momentum * liquidity
    
    return factor
