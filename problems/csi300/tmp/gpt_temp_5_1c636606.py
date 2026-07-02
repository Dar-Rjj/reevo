import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Intraday Momentum Component
    df['intraday_momentum'] = (df['high'] - df['low']) / df['low']
    
    # Rolling Momentum Component
    df['close_to_close_return'] = df['close'].pct_change()
    df['rolling_momentum'] = df['close_to_close_return'].rolling(window=20, min_periods=1).mean()
    
    # Find Last Peak Close Price in Last 20 Days
    df['rolling_max_close'] = df['close'].rolling(window=20, min_periods=1).max()
    df['days_since_last_peak'] = df.apply(lambda row: (df.loc[:row.name, 'close'] == row['rolling_max_close']).idxmax(), axis=1)
    df['days_since_last_peak'] = df.index.get_indexer(df['days_since_last_peak'])
    
    # Multiply by Days Since Last Peak
    df['momentum_adjusted'] = df['rolling_momentum'] * df['days_since_last_peak']
    
    # Volume-based Reversal
    df['rolling_volume_mean'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['normalized_volume'] = df['volume'] / df['rolling_volume_mean']
    df['reversal'] = -df['momentum_adjusted'] * df['normalized_volume']
    
    # Range Adjustment
    df['intraday_range'] = df['high'] - df['low']
    df['rolling_range_mean'] = df['intraday_range'].rolling(window=5, min_periods=1).mean()
    df['range_adjustment'] = df['rolling_range_mean'] / df['close']
    
    # Multiply Reversal by Range Adjustment
    df['final_factor'] = df['reversal'] * df['range_adjustment']
    
    return df['final_factor']
