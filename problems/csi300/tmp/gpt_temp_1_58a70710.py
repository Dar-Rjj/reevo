def heuristics_v2(df):
    # Price Momentum Component
    # Short-Term Price Trend
    ma5 = df['close'].rolling(5).mean()
    ma20 = df['close'].rolling(20).mean()
    
    # Momentum Strength
    ma5_change = ma5.diff()
    ma20_change = ma20.diff()
    
    # Volume Divergence Component
    # Volume Anomaly
    rolling_vol_mean = df['volume'].rolling(20).mean()
    volume_anomaly = df['volume'] / rolling_vol_mean
    
    # Volume Z-score
    rolling_vol_std = df['volume'].rolling(20).std()
    volume_zscore = (df['volume'] - rolling_vol_mean) / rolling_vol_std
    
    # Combine Signals
    momentum_signal = ma5_change - ma20_change
    combined_signal = momentum_signal * volume_zscore
    
    # Final normalization
    factor = combined_signal.rolling(5).apply(
        lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    return factor
