import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original DataFrame
    data = df.copy()
    
    # 1. Measure Price Range Expansion
    # Current day's range
    daily_range = (data['high'] - data['low']) / data['close']
    
    # 10-day moving average of daily ranges (using only past data)
    range_ma = daily_range.rolling(window=10, min_periods=1).mean()
    
    # Range expansion ratio (current range vs historical average)
    range_ratio = daily_range / range_ma
    
    # 2. Volume Confirmation
    # Calculate volume percentile (0-1) over 10-day lookback
    volume_percentile = data['volume'].rolling(window=10, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    
    # Apply logistic scaling to volume percentile
    volume_score = 1 / (1 + np.exp(-10 * (volume_percentile - 0.5)))
    
    # 3. Price Confirmation
    # Compute intraday price strength
    price_strength = (data['close'] - data['open']) / data['close']
    
    # Normalize price strength using 10-day standard deviation
    price_std = price_strength.rolling(window=10, min_periods=1).std()
    normalized_price = price_strength / price_std.replace(0, 1)  # avoid division by zero
    
    # Combine all signals multiplicatively
    factor = range_ratio * volume_score * normalized_price
    
    return factor
