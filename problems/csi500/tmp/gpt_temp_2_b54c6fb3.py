def heuristics_v2(df):
    # Calculate Buy Pressure
    buy_pressure = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Calculate Sell Pressure
    sell_pressure = (df['open'] - df['close']) / (df['high'] - df['low'])
    
    # Compute Net Imbalance
    net_imbalance = buy_pressure - sell_pressure
    
    # Compute Volume Acceleration
    volume_ma_5 = df['volume'].rolling(window=5, min_periods=1).mean()
    volume_momentum = (df['volume'] - volume_ma_5) / volume_ma_5
    
    # Combine Imbalance and Momentum
    combined_signal = net_imbalance * volume_momentum
    
    # Normalize with 5-day Z-score
    z_score = combined_signal.rolling(window=5, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std(), raw=False
    )
    
    return z_score
