def heuristics_v2(df):
    # Calculate intraday reversal component
    intraday_reversal = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-6)
    
    # Normalize reversal by 5-day rolling standard deviation (using only past data)
    reversal_std = intraday_reversal.rolling(5, min_periods=3).std()
    normalized_reversal = intraday_reversal / (reversal_std + 1e-6)
    
    # Calculate liquidity filters
    sma_volume = df['volume'].rolling(20, min_periods=10).mean()
    sma_amount = df['amount'].rolling(20, min_periods=10).mean()
    
    volume_filter = df['volume'] > sma_volume
    amount_filter = df['amount'] > sma_amount
    
    # Combine signals - only apply reversal when both liquidity filters are true
    combined_signal = normalized_reversal.where(volume_filter & amount_filter, 0)
    
    # Rank normalization
    ranked_signal = combined_signal.rolling(20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] - 0.5) if len(x.dropna()) >= 10 else 0
    )
    
    return ranked_signal
