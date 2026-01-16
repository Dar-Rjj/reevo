def heuristics_v2(df):
    # Calculate current range ratio
    current_range = (df['high'] - df['low']) / df['close']
    prev_range = current_range.shift(1)
    range_ratio = current_range / prev_range
    
    # Calculate midpoint deviation
    midpoint = (df['high'] + df['low']) / 2
    midpoint_deviation = (df['close'] - midpoint) / (df['high'] - df['low'])
    
    # Combine to create reversal signal
    reversal_signal = midpoint_deviation * range_ratio
    
    # Calculate volume ratio (current volume / 5-day average volume)
    volume_avg = df['volume'].rolling(5).mean()
    volume_ratio = df['volume'] / volume_avg
    
    # Create final signal weighted by volume
    factor = reversal_signal * volume_ratio
    
    # Cross-sectional normalization
    def normalize(series):
        mean = series.mean()
        std = series.std()
        if std == 0:
            return series * 0  # return zeros if no variation
        normalized = (series - mean) / std
        return normalized.clip(-3, 3)
    
    # Apply normalization within each day
    factor = factor.groupby(factor.index).transform(normalize)
    
    return factor
