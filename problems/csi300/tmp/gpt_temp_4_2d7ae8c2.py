import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Normalized Range
    df['range'] = df['high'] - df['low']
    df['normalized_range'] = df['range'] / df['close'].shift(1)

    # Compute 5-day ATR
    df['tr'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - df['close'].shift(1).loc[x.name]), abs(df['close'].shift(1).loc[x.name] - x['low'])),
        axis=1
    )
    df['atr'] = df['tr'].rolling(window=5).mean()

    # Compare Current Range to ATR
    df['range_over_atr'] = df['normalized_range'] / df['atr']

    # Calculate Volume Surprise
    df['volume_median'] = df['volume'].rolling(window=10).median()
    df['volume_surprise'] = df['volume'] / df['volume_median']

    # Combine Components
    df['factor'] = df['range_over_atr'] * df['volume_surprise'] * (df['close'] - df['open']).apply(lambda x: 1 if x > 0 else -1)

    return df['factor']
