def heuristics_v2(df):
    # Calculate midpoint price
    midpoint = (df['high'] + df['low']) / 2
    
    # Price divergence component
    price_divergence = (df['close'] - midpoint) / midpoint
    # Standardize using 10-day rolling z-score (historical only)
    price_div_z = price_divergence.rolling(window=10, min_periods=1).apply(
        lambda x: (x[-1] - x[:-1].mean()) / x[:-1].std() if x[:-1].std() != 0 else 0
    )
    
    # Volume momentum component
    vol_ma5 = df['volume'].rolling(window=5, min_periods=1).mean()
    vol_ratio = df['volume'] / vol_ma5
    # Clip volume ratio to [0.5, 2.0]
    vol_ratio_clipped = vol_ratio.clip(lower=0.5, upper=2.0)
    
    # Combined signal
    combined = price_div_z * vol_ratio_clipped
    
    # Cross-sectional normalization
    def cross_section_normalize(series):
        mean = series.mean()
        std = series.std()
        if std != 0:
            return (series - mean) / std
        return 0
    
    # Apply cross-sectional normalization and clip
    normalized = combined.groupby(combined.index).transform(cross_section_normalize)
    normalized_clipped = normalized.clip(lower=-3, upper=3)
    
    return normalized_clipped
