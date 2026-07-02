def heuristics_v2(df):
    # Momentum Divergence Component
    rolling_mean_close = df['close'].rolling(window=15, min_periods=1).mean()
    relative_strength = df['close'] / rolling_mean_close
    delta_relative_strength = relative_strength.diff(periods=5)
    rolling_std_close = df['close'].rolling(window=15, min_periods=1).std()
    normalization = delta_relative_strength / rolling_std_close
    momentum_divergence_component = normalization.abs()
    
    # Liquidity Weighting
    rolling_mean_volume = df['volume'].rolling(window=15, min_periods=1).mean()
    normalize_volume = df['volume'] / rolling_mean_volume
    microstructure_impact = (df['amount'] / df['volume']).rolling(window=15, min_periods=1).apply(lambda x: x.rank().iloc[-1])
    liquidity_weighting = normalize_volume * microstructure_impact
    
    # Decay Factor
    decay_factor = liquidity_weighting.ewm(alpha=0.5, adjust=False).mean().rolling(window=3, min_periods=1).mean()
    
    # Factor Calculation
    factor = momentum_divergence_component * decay_factor
    
    return factor
