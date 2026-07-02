import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy of the dataframe to avoid modifying the original
    data = df.copy()
    
    # Price Momentum Component
    # 1. Calculate EMA(close, span=12)
    ema = data['close'].ewm(span=12, adjust=False).mean()
    
    # 2. Calculate returns
    returns = data['close'].pct_change()
    
    # 3. Calculate rolling rank of returns over 20 days
    rolling_rank = returns.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Combine momentum components
    momentum = ema * rolling_rank
    
    # Liquidity Confirmation
    # 1. Calculate ratio of current amount to 10-day rolling mean of amount
    rolling_mean_amount = data['amount'].rolling(window=10).mean()
    amount_ratio = data['amount'] / rolling_mean_amount
    
    # 2. Normalize cross-sectional liquidity (z-score)
    # Using current day's amount divided by its 20-day rolling std
    liquidity_zscore = (data['amount'] - data['amount'].rolling(window=20).mean()) / data['amount'].rolling(window=20).std()
    
    # Combine liquidity components
    liquidity = amount_ratio * liquidity_zscore
    
    # Final factor: combine momentum and liquidity components
    factor = momentum * liquidity
    
    # Return as a pandas Series with the same index as input
    return pd.Series(factor, index=data.index)
