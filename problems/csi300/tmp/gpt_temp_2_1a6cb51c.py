import pandas as pd
def heuristics_v2(data):
    # Calculate intraday range (high - low) / close
    intraday_range = (data['high'] - data['low']) / data['close']
    
    # Calculate 5-day rolling median range
    median_range = intraday_range.rolling(5).median()
    
    # Compare current range to historical range
    range_ratio = intraday_range / median_range
    
    # Calculate close-to-close returns
    returns = data['close'].pct_change()
    
    # Calculate 5-day average volume
    avg_volume = data['volume'].rolling(5).mean()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for t in data.index:
        if range_ratio.loc[t] > 2:
            # Abnormal movement: use negative returns for reversion
            factor.loc[t] = -returns.loc[t]
        else:
            # Normal movement: volume-adjusted returns
            if avg_volume.loc[t] > 0:
                factor.loc[t] = returns.loc[t] / avg_volume.loc[t]
            else:
                factor.loc[t] = 0
    
    return factor
