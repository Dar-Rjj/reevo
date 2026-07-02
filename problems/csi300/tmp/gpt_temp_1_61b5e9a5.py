def heuristics_v2(df):
    # Measure Price Efficiency
    intraday_range = df['high'] - df['low']
    closing_efficiency = (df['close'] - df['low']) / intraday_range.replace(0, 1e-6)  # Avoid division by zero
    
    # Measure Volume Stability
    rolling_volume_std = df['volume'].rolling(window=10, min_periods=1).std()
    volume_stability = 1 / (1 + rolling_volume_std)  # Inverse relationship with stability
    
    # Combine Signals
    combined_signal = closing_efficiency * volume_stability
    
    # Apply Z-Score normalization
    z_score = combined_signal.rolling(window=5, min_periods=1).apply(
        lambda x: (x[-1] - x[:-1].mean()) / (x[:-1].std() + 1e-6)
    )
    
    return z_score
