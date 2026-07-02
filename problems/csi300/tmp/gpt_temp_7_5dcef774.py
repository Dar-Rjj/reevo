def heuristics_v2(df):
    # Calculate Short-Term Momentum
    short_term_momentum = df['close'].rolling(window=5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], raw=False)
    
    # Calculate Long-Term Momentum
    long_term_momentum = df['close'].rolling(window=20).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], raw=False)
    
    # Calculate Divergence from Short-Term Momentum
    divergence = short_term_momentum - long_term_momentum
    
    # Normalize by Long-Term Momentum Standard Deviation
    long_term_std = long_term_momentum.rolling(window=20).std()
    normalized_divergence = divergence / long_term_std
    
    return normalized_divergence
