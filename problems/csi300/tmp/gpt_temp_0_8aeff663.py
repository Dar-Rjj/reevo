def heuristics_v2(df):
    # Calculate Price Change
    df['price_change'] = (df['close'] - df['low']) / df['close']
    df['abs_price_change'] = df['price_change'].abs()
    
    # Normalized Range
    df['norm_range'] = df['high'] / df['low']
    df['abs_norm_range'] = df['norm_range'].abs()
    
    # Volume Confirmation
    df['10_day_ma_volume'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['10_day_ma_volume']
    
    # Volatility Adjustment
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda row: max(row['high'] - row['low'], row['high'] - df['close'].shift(1).loc[row.name], df['close'].shift(1).loc[row.name] - row['low']),
        axis=1
    )
    df['5_day_ma_true_range'] = df['true_range'].rolling(window=5, min_periods=1).mean()
    
    # Detect Overreaction and Generate Final Signal
    df['overreaction'] = df['true_range'] > 1.5 * df['5_day_ma_true_range']
    df['final_signal'] = df.apply(
        lambda row: (row['price_change'] * row['volume_ratio']) / row['true_range'] if row['overreaction'] else 0,
        axis=1
    )
    
    # Cap at ±2 standard deviations of historical Price Change
    price_change_std = df['price_change'].std()
    df['final_signal'] = df['final_signal'].clip(lower=-2 * price_change_std, upper=2 * price_change_std)
    
    return df['final_signal']
