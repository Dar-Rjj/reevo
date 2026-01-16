def heuristics_v2(df):
    # Calculate price reversal signal
    price_deviation = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    reversal_signal = -1 * price_deviation
    
    # Clip extreme values using rolling statistics (3 std dev)
    rolling_std = reversal_signal.expanding().std()
    mean_signal = reversal_signal.expanding().mean()
    upper_bound = mean_signal + 3 * rolling_std
    lower_bound = mean_signal - 3 * rolling_std
    reversal_signal = reversal_signal.clip(lower_bound, upper_bound)
    
    # Calculate normalized volume
    rolling_volume_mean = df['volume'].rolling(window=20, min_periods=1).mean()
    normalized_volume = df['volume'] / rolling_volume_mean
    
    # Clip normalized volume at 3 standard deviations
    vol_rolling_std = normalized_volume.expanding().std()
    vol_mean = normalized_volume.expanding().mean()
    vol_upper = vol_mean + 3 * vol_rolling_std
    vol_lower = vol_mean - 3 * vol_rolling_std
    normalized_volume = normalized_volume.clip(vol_lower, vol_upper)
    
    # Combine signals and calculate cross-sectional z-score
    combined_signal = reversal_signal * normalized_volume
    factor = combined_signal.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())
    
    return factor
