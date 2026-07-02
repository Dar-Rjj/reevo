import pandas as pd
def heuristics_v2(df):
    # Initialize output series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Calculate intraday momentum components
    df['high_close_ratio'] = (df['high'] - df['close']) / df['close']
    df['low_close_ratio'] = (df['low'] - df['close']) / df['close']
    
    # Calculate overreaction detection components
    df['prev_close'] = df['close'].shift(1)
    df['morning_gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    
    df['high_open_ratio'] = (df['high'] - df['open']) / df['open']
    df['low_open_ratio'] = (df['low'] - df['open']) / df['open']
    
    # Calculate factor values
    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Intraday Momentum
        intraday_momentum = (current['high_close_ratio'] + current['low_close_ratio']) / 2
        
        # Overreaction Detection
        overreaction = 0
        if abs(current['morning_gap']) > 0.02:  # 2% threshold
            if current['morning_gap'] > 0:  # Positive gap (potential overreaction)
                overreaction = -current['high_open_ratio']  # Expect reversal from highs
            else:  # Negative gap (potential overreaction)
                overreaction = -current['low_open_ratio']  # Expect reversal from lows
        
        # Combine components
        factor_values.iloc[i] = intraday_momentum + overreaction
    
    return factor_values
