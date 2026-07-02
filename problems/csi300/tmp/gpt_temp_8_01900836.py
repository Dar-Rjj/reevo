import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Liquidity Change Signal
    # Calculate rolling amount delta over 3 days
    amount_delta = df['amount'].diff(periods=3)
    
    # Calculate rolling rank of the delta over 15 days
    liquidity_rank = amount_delta.rolling(window=15).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Calculate cross-sectional rank and z-score
    liquidity_zscore = liquidity_rank.groupby(level=0).apply(lambda x: (x - x.mean()) / x.std())
    
    # Price Momentum Confirmation
    # Calculate EMA of close prices with span 5
    ema_close = df['close'].ewm(span=5, adjust=False).mean()
    
    # Calculate returns over 2 days
    returns = df['close'].pct_change(periods=2)
    
    # Calculate rolling standard deviation of returns over 5 days
    vol = returns.rolling(window=5).std()
    
    # Calculate volatility-adjusted returns
    vol_adj_returns = returns / vol
    
    # Combine signals
    factor = liquidity_zscore + ema_close.pct_change() + vol_adj_returns
    
    return factor
