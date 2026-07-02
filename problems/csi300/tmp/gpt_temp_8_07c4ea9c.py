import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Efficiency component
    # Ratio of high to low prices
    price_ratio = df['high'] / df['low']
    
    # Rolling rank of the price ratio (window=10)
    rolling_rank = price_ratio.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Cross-sectional rank normalization
    normalized_rank = rolling_rank.groupby(rolling_rank.index).rank(pct=True)
    
    # Order Flow Asymmetry component
    # Volume delta (current vs rolling mean of past 5 days)
    volume_delta = df['volume'] - df['volume'].shift(5).rolling(10).mean()
    
    # Correlation between price changes and amount changes
    price_delta = df['close'] - df['close'].shift(1)
    amount_delta = df['amount'] - df['amount'].shift(1)
    
    # Rolling correlation (window=10) between price_delta and amount_delta
    rolling_corr = pd.Series(
        [price_delta.rolling(10).corr(amount_delta).iloc[i] for i in range(len(price_delta))],
        index=price_delta.index
    )
    
    # Combine components with equal weights
    factor = 0.5 * normalized_rank + 0.5 * rolling_corr
    
    return factor
