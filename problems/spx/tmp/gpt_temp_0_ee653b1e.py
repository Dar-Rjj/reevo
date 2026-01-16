def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute rolling average of Intraday Range
    df['rolling_avg_range'] = df['intraday_range'].rolling(window=10, min_periods=1).mean()
    
    # Intraday Range Momentum: Subtract rolling average from current Intraday Range
    df['range_momentum'] = df['intraday_range'] - df['rolling_avg_range']
    
    # Compute Volume Trend: Rolling Volume Slope
    df['volume_slope'] = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: (x[-1] - x[0]) / len(x) if len(x) > 1 else 0)
    
    # Sign of Volume Slope
    df['volume_trend'] = df['volume_slope'].apply(lambda x: 1 if x > 0 else -1)
    
    # Scale Momentum by Volume Trend and take Square Root
    df['volume_adjusted_momentum'] = df['range_momentum'] * df['volume_trend']
    df['factor'] = df['volume_adjusted_momentum'].apply(lambda x: x**0.5 if x > 0 else -(-x)**0.5)
    
    return df['factor']
