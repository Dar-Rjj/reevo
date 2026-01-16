def heuristics_v2(df):
    # Price Component
    df['short_term_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['long_term_momentum'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Volume Component
    df['volume_percentile'] = df['volume'].rolling(window=20, min_periods=1).apply(lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-9))
    df['volume_adjusted_signal'] = (df['short_term_momentum'] + df['long_term_momentum']) * df['volume_percentile']
    
    # Range Component
    df['intraday_range'] = (df['high'] - df['low']) / df['open']
    df['final_factor'] = df['volume_adjusted_signal'] * df['intraday_range']
    
    return df['final_factor']
