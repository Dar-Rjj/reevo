import pandas as pd
def heuristics_v2(df):
    # Calculate overnight price gap
    df['gap'] = df['open'] / df['close'].shift(1) - 1
    df['abs_gap'] = df['gap'].abs()
    
    # Calculate volume moving average
    df['ma_volume'] = df['volume'].rolling(window=20).mean()
    
    # Determine liquidity condition
    df['high_liquidity'] = df['volume'] > df['ma_volume']
    
    # Calculate momentum for high liquidity periods
    df['momentum'] = df['gap'].rolling(window=5).mean()
    
    # Initialize factor values
    factor_values = pd.Series(0, index=df.index)
    
    # Apply rules based on the tree
    for i in range(1, len(df)):
        # Skip if we don't have enough history
        if pd.isna(df.iloc[i]['gap']) or pd.isna(df.iloc[i]['ma_volume']):
            continue
            
        gap = df.iloc[i]['gap']
        abs_gap = df.iloc[i]['abs_gap']
        high_liq = df.iloc[i]['high_liquidity']
        momentum = df.iloc[i]['momentum']
        
        # Only proceed if we have high liquidity
        if high_liq:
            if gap > 0:  # Positive gap
                factor_values.iloc[i] = -momentum * abs_gap  # Reversal signal
            elif gap < 0:  # Negative gap
                factor_values.iloc[i] = momentum * abs_gap   # Reversal signal
    
    return factor_values
