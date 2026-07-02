import pandas as pd
def heuristics_v2(df):
    # Calculate overnight price gap
    df['gap'] = df['open'] / df['close'].shift(1) - 1
    df['abs_gap'] = df['gap'].abs()
    
    # Calculate 10-day moving average of volume (using only past data)
    df['ma_volume'] = df['volume'].rolling(window=10, min_periods=1).mean()
    
    # Initialize factor series
    factor = pd.Series(0, index=df.index)
    
    for t in df.index:
        # Skip first day (no previous close)
        if pd.isna(df.loc[t, 'gap']):
            continue
            
        gap = df.loc[t, 'gap']
        abs_gap = df.loc[t, 'abs_gap']
        volume = df.loc[t, 'volume']
        ma_volume = df.loc[t, 'ma_volume']
        
        # Check liquidity condition
        high_liquidity = volume > 1.5 * ma_volume
        
        # Check gap size
        large_gap = abs_gap > 0.01
        
        # Calculate factor value
        if large_gap and high_liquidity:
            factor.loc[t] = gap * volume / ma_volume
        else:
            factor.loc[t] = 0
            
    return factor
