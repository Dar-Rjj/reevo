import pandas as pd
def heuristics_v2(data):
    """
    Calculate Intraday Range-Volume Efficiency factor based on price range and volume signals.
    
    Parameters:
    data (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    
    Returns:
    pd.Series: Factor values indexed by date
    """
    # Calculate normalized range (High - Low)/Close
    normalized_range = (data['high'] - data['low']) / data['close']
    
    # Calculate 10-day moving average of normalized ranges (expected range)
    expected_range = normalized_range.rolling(window=10, min_periods=1).mean()
    
    # Price Range Efficiency: current range vs expected range
    range_efficiency = (normalized_range - expected_range) / expected_range
    
    # Calculate volume percentile (0-1) over past 20 days
    def rolling_percentile(s):
        return s.rolling(window=20, min_periods=1).apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
    
    volume_percentile = rolling_percentile(data['volume'])
    
    # Combine signals: Range Efficiency * Volume Percentile
    factor = range_efficiency * volume_percentile
    
    return factor
