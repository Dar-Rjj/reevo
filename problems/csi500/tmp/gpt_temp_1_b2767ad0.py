import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate midpoint price
    midpoint = (df['high'] + df['low']) / 2
    
    # Calculate price deviation from midpoint
    price_deviation = df['close'] - midpoint
    
    # Calculate daily range
    daily_range = df['high'] - df['low']
    daily_range = daily_range.replace(0, np.nan)  # Avoid division by zero
    
    # Normalize price deviation by daily range
    normalized_deviation = price_deviation / daily_range
    
    # Calculate volume percentile over 5-day rolling window
    volume_percentile = df['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Calculate amount-based liquidity
    amount_liquidity = df['amount'] / daily_range
    amount_liquidity = amount_liquidity.replace([np.inf, -np.inf], np.nan)
    
    # Combine reversal signal with liquidity filters
    factor = normalized_deviation * volume_percentile * amount_liquidity
    
    return factor
