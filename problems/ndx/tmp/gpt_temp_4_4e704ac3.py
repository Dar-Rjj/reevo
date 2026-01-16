import numpy as np
def heuristics_v2(df):
    # Compute normalized range
    normalized_range = (df['high'] - df['low']) / df['close']
    
    # Compute price direction
    price_direction = (df['close'] - df['open']).apply(np.sign)
    
    # Calculate intraday momentum
    intraday_momentum = normalized_range * price_direction
    
    # Calculate volume spike (current volume / 10-day average volume)
    avg_volume_10day = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_spike = df['volume'] / avg_volume_10day
    
    # Weight momentum by volume confirmation
    volume_adjusted_momentum = intraday_momentum * volume_spike
    
    # Cross-sectional normalization
    def zscore(series):
        return (series - series.mean()) / series.std()
    
    # Apply cross-sectional z-score and clip values
    factor = volume_adjusted_momentum.groupby(volume_adjusted_momentum.index).transform(zscore)
    factor = factor.clip(-2, 2)
    
    return factor
