import numpy as np
def heuristics_v2(df):
    # Calculate daily price range
    daily_range = df['high'] - df['low']
    
    # Calculate average price
    avg_price = (df['high'] + df['low']) / 2
    
    # Normalize daily range by average price
    normalized_range = daily_range / avg_price
    
    # Calculate volume-weighted rolling mean over 5 days
    def weighted_mean(x):
        weights = df['volume'].loc[x.index]
        return (x * weights).sum() / weights.sum()
    
    smoothed_range = normalized_range.rolling(5).apply(weighted_mean, raw=False)
    
    # Calculate log volume ratio
    rolling_mean_volume = df['volume'].rolling(5).mean()
    log_volume_ratio = np.log(df['volume'] / rolling_mean_volume)
    
    # Final factor calculation
    factor = smoothed_range * log_volume_ratio
    
    return factor
