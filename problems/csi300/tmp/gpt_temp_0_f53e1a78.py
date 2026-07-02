def heuristics_v2(df):
    # Calculate Close-to-Open Ratio
    close_to_open_ratio = df['close'] / df['open']
    
    # Normalize by Intraday Range
    intraday_range = df['high'] - df['low']
    price_efficiency = close_to_open_ratio / intraday_range
    
    # Calculate Volume Percentile
    volume_percentile = df['volume'].rolling(window=20).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    # Combine Signals
    factor = price_efficiency * volume_percentile
    
    return factor
