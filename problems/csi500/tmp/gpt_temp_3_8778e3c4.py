import pandas as pd
def heuristics_v2(df):
    """
    Calculate Relative Money Flow Index (RMFI) based on typical price and money flow.
    Rules followed:
    1. Only uses current and past data (no future information)
    2. No negative shifts or forward-looking
    3. Rolling calculations use only historical data
    """
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate money flow
    money_flow = typical_price * df['volume']
    
    # Initialize positive and negative money flow
    positive_mf = pd.Series(0, index=df.index)
    negative_mf = pd.Series(0, index=df.index)
    
    # Calculate positive and negative money flow
    for i in range(1, len(df)):
        if typical_price[i] > typical_price[i-1]:
            positive_mf[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_mf[i] = money_flow[i]
    
    # Calculate cumulative sums for 14 periods (standard RMFI period)
    cum_pos_mf = positive_mf.rolling(window=14, min_periods=1).sum()
    cum_neg_mf = negative_mf.rolling(window=14, min_periods=1).sum()
    
    # Calculate money flow ratio
    mf_ratio = cum_pos_mf / cum_neg_mf
    
    # Calculate RMFI
    rmfi = 100 - (100 / (1 + mf_ratio))
    
    return rmfi
