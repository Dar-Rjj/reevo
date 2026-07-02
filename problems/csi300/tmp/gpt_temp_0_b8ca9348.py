def heuristics_v2(df):  
    # Calculate rolling mean of price returns (5-day)  
    rolling_mean_returns = df['close'].pct_change().rolling(5).mean()  
      
    # Calculate rolling skewness of volume (5-day)  
    rolling_skewness_volume = df['volume'].rolling(5).apply(lambda x: x.skew())  
      
    # Combine rolling mean returns (rank-weighted) and rolling skewness volume (rank-weighted)  
    heuristics_matrix = (rolling_mean_returns.rank(pct=True) * 0.5 + rolling_skewness_volume.rank(pct=True) * 0.5)  
      
    return heuristics_matrix
