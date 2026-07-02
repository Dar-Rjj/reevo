import pandas as pd
import pandas as pd

def heuristics_v2(data):
    # Calculate Intraday Price Range
    data['range'] = data['high'] - data['low']
    
    # Normalize Range by Open Price
    data['normalized_range'] = data['range'] / data['open']
    
    # Calculate Rolling 5-Day Average Range
    data['rolling_avg_range'] = data['range'].rolling(window=5, min_periods=1).mean()
    
    # Compute Current Range Percentile within 20-Day Range Distribution
    data['range_percentile'] = data['range'].rolling(window=20, min_periods=1).apply(
        lambda x: pd.Series(x[-1]).rank(pct=True).iloc[0] * 100
    )
    
    # Normalize Percentile
    data['normalized_percentile'] = data['range_percentile'] / 100
    
    # Intraday Range Momentum with Historical Normalization
    momentum_factor = data['normalized_range'] * data['normalized_percentile']
    
    return momentum_factor
