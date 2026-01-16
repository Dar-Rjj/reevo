def heuristics_v2(df):
    # Intraday Rejection Signal
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'])
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'])
    
    # Momentum Integration
    df['intraday_momentum'] = (df['close'] - df['open']) / df['open']
    df['upper_shadow_momentum'] = df['upper_shadow'] * df['intraday_momentum']
    df['lower_shadow_momentum'] = df['lower_shadow'] * df['intraday_momentum']
    
    # Volume Adjustment
    df['volume_ma5'] = df['volume'].rolling(window=5, min_periods=1).mean()
    df['volume_spike'] = df['volume'] / df['volume_ma5']
    df['weighted_upper_shadow_momentum'] = df['upper_shadow_momentum'] * df['volume_spike']
    df['weighted_lower_shadow_momentum'] = df['lower_shadow_momentum'] * df['volume_spike']
    
    # Signal Processing
    df['net_signal'] = df['weighted_upper_shadow_momentum'] - df['weighted_lower_shadow_momentum']
    df['normalized_signal'] = df['net_signal'].rolling(window=5, min_periods=1).apply(lambda x: (x.rank(pct=True).iloc[-1]), raw=False)
    
    return df['normalized_signal']
