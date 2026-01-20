def heuristics_v2(df):
    # Range-Adjusted Momentum Component
    # Calculate Intraday Momentum
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-6)
    
    # Compare to Historical Momentum
    momentum_sma = intraday_momentum.rolling(5).mean()
    momentum_divergence = intraday_momentum - momentum_sma
    
    # Liquidity Adjustment Component
    # Compute Volume Ratio
    volume_sma = df['volume'].rolling(10).mean()
    volume_ratio = df['volume'] / volume_sma.replace(0, 1e-6)
    
    # Apply Smoothing
    liquidity_adjustment = volume_ratio.ewm(span=3, adjust=False).mean()
    
    # Combined Signal
    combined_signal = momentum_divergence * liquidity_adjustment
    
    # Apply Z-Score normalization
    z_score = combined_signal.rolling(5).apply(
        lambda x: (x[-1] - x[:-1].mean()) / x[:-1].std() if x[:-1].std() != 0 else 0
    )
    
    return z_score
